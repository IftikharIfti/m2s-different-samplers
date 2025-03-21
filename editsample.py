import argparse
import os
import time
import numpy as np
from PIL import Image
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
from metrics_cal import *
import math

def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs

def create_model(image_size, my_args):
    args = argparse.Namespace(**args_to_dict(my_args, model_and_diffusion_defaults().keys()))
    args.image_size = image_size
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    return model, diffusion

def upsample_image(image_tensor, target_size=(256, 256)):
    image_tensor = (image_tensor + 1) / 2.0
    image_tensor = image_tensor.clamp(0, 1)
    upscaled_image_tensor = F.interpolate(image_tensor, size=target_size, mode="bilinear", align_corners=False)
    upscaled_image_tensor = upscaled_image_tensor * 2.0 - 1
    return upscaled_image_tensor

def main():
    start_time = time.time()
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir, format_strs=['stdout'])  # Ensure console output

    logger.log("creating model...")
    if args.use_cf:
        model_64, diffusion_64 = create_model(image_size=64, my_args=args)
        model_64.load_state_dict(dist_util.load_state_dict(args.model_path_64, map_location="cpu"))
        model_64.to(dist_util.dev())
        if args.use_fp16:
            model_64.convert_to_fp16()
        model_64.eval()
    
    model_256, diffusion_256 = create_model(image_size=256, my_args=args)
    model_256.load_state_dict(dist_util.load_state_dict(args.model_path_256, map_location="cpu"))
    model_256.to(dist_util.dev())
    if args.use_fp16:
        model_256.convert_to_fp16()
    model_256.eval()

    logger.log("loading data...")
    try:
        data_64 = load_reference(args.base_samples, args.batch_size, image_size=64, class_cond=args.class_cond)
        data_mask_64 = load_reference(args.mask_path, args.batch_size, image_size=64, class_cond=args.class_cond)
        data_256 = load_reference(args.base_samples, args.batch_size, image_size=256, class_cond=args.class_cond)
        data_mask_256 = load_reference(args.mask_path, args.batch_size, image_size=256, class_cond=args.class_cond)
    except Exception as e:
        logger.log(f"Error loading data: {str(e)}")
        raise

    logger.log("creating samples...")
    count = 0
    all_items = os.listdir(args.base_samples)
    num_inputs = len(all_items)
    logger.log(f"Number of inputs: {num_inputs}")

    metrics_file_path = os.path.join(logger.get_dir(), "metrics_log.txt")
    lpips_value = 0.
    coarse_lpips_value = 0.
    psnr_value = 0.
    coarse_psnr_value = 0.
    ssim_value = 0.
    coarse_ssim_value = 0.
    l1_value = 0.
    coarse_l1_value = 0.
    
    with open(metrics_file_path, "a") as metrics_file:
        metrics_file.write(f"Condition:\n")
        metrics_file.write(f"\tmask_path: {args.mask_path}\n")
        metrics_file.write(f"\tn_sample: {args.n_sample}\n")
        metrics_file.write(f"\tuse_cf: {args.use_cf}\n")
        metrics_file.write(f"\tspecial_mask: {args.special_mask}\n")
        if args.use_cf:
            metrics_file.write(f"\tt_T_coarse: {args.t_T}\n")
            metrics_file.write(f"\tt_T_fine: {args.t_T_fine}\n")
            metrics_file.write(f"\tdpm_steps_coarse: {args.dpm_steps_coarse}\n")
            metrics_file.write(f"\tdpm_steps_fine: {args.dpm_steps_fine}\n")
        else:
            metrics_file.write(f"\tt_T: {args.t_T}\n")
            metrics_file.write(f"\tdpm_steps: {args.dpm_steps}\n")
        metrics_file.write(f"\n")
    
    while count < num_inputs:
        try:
            logger.log("Fetching data batch...")
            model_kwargs_64 = next(data_64)
            model_mask_kwargs_64 = next(data_mask_64)
            model_kwargs_256 = next(data_256)
            model_mask_kwargs_256 = next(data_mask_256)
            logger.log("Data batch fetched.")
            
            model_kwargs_64 = {k: v.to(dist_util.dev()) for k, v in model_kwargs_64.items()}
            model_mask_kwargs_64 = {k: v.to(dist_util.dev()) for k, v in model_mask_kwargs_64.items()}
            model_kwargs_256 = {k: v.to(dist_util.dev()) for k, v in model_kwargs_256.items()}
            model_mask_kwargs_256 = {k: v.to(dist_util.dev()) for k, v in model_mask_kwargs_256.items()}
            
            gt = model_kwargs_256["ref_img"]
            
            if args.use_inverse_masks:
                model_mask_kwargs_64["ref_img"] = model_mask_kwargs_64["ref_img"] * (-1)
                model_mask_kwargs_256["ref_img"] = model_mask_kwargs_256["ref_img"] * (-1)
            
            mask = model_mask_kwargs_256["ref_img"]
            mask[mask < 0.] = 0
            mask[mask > 0.] = 1
            
            noise = None
            t_T_fine = args.t_T_fine if args.use_cf else args.t_T
            model_fine_kwargs = model_kwargs_256
            cond_fn = None
            
            if args.use_cf:
                logger.log("Starting coarse sampling...")
                if args.special_mask == False:
                    sample_coarse_iter = diffusion_64.p_sample_loop_progressive(
                        model_64,
                        (args.batch_size, 3, 64, 64),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs_64,
                        model_mask_kwargs=model_mask_kwargs_64,
                        progress=True,
                        dpm_steps=args.dpm_steps_coarse,
                    )
                    sample_coarse = list(sample_coarse_iter)[-1]["sample"]
                    logger.log("Coarse sampling completed.")
                    t_T_tensor = th.tensor([t_T_fine] * sample_coarse.shape[0]).to(sample_coarse.device)
                    sample_coarse_256 = upsample_image(sample_coarse)
                else:
                    sample_coarse_iter = diffusion_256.p_sample_loop_progressive(
                        model_256,
                        (args.batch_size, 3, 256, 256),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs_256,
                        model_mask_kwargs=model_mask_kwargs_256,
                        progress=True,
                        dpm_steps=args.dpm_steps_coarse,
                    )
                    sample_coarse = list(sample_coarse_iter)[-1]["sample"]
                    logger.log("Coarse sampling completed.")
                    t_T_tensor = th.tensor([t_T_fine] * sample_coarse.shape[0]).to(sample_coarse.device)
                    sample_coarse_256 = sample_coarse
                
                sample_coarse = th.clamp(sample_coarse, -1.0, 1.0)
                noise = diffusion_256.q_sample(sample_coarse_256, t_T_tensor)
                model_fine_kwargs["ref_img"] = sample_coarse_256 * (1 - mask) + model_kwargs_256["ref_img"] * mask
            
            logger.log("Starting fine sampling...")
            sample_fine_iter = diffusion_256.p_sample_loop_progressive(
                model_256,
                (args.batch_size, 3, 256, 256),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_fine_kwargs,
                model_mask_kwargs=model_mask_kwargs_256,
                progress=True,
                dpm_steps=args.dpm_steps_fine if args.use_cf else args.dpm_steps,
                noise=noise,
            )
            sample_fine = list(sample_fine_iter)[-1]["sample"]
            logger.log("sample_fine completed.")
            sample_fine = th.clamp(sample_fine, -1.0, 1.0)
            
            # Rest of the processing...
            for i in range(args.batch_size):
                os.makedirs(os.path.join(logger.get_dir(), "gtImg"), exist_ok=True)
                os.makedirs(os.path.join(logger.get_dir(), "inputImg"), exist_ok=True)
                os.makedirs(os.path.join(logger.get_dir(), "sampledImg"), exist_ok=True)
                os.makedirs(os.path.join(logger.get_dir(), "outImg"), exist_ok=True)
                os.makedirs(os.path.join(logger.get_dir(), "coarseImg"), exist_ok=True)
                
                out_gtImg_path = os.path.join(logger.get_dir(), "gtImg", f"{str(count + i).zfill(4)}.png")
                out_inputImg_path = os.path.join(logger.get_dir(), "inputImg", f"{str(count + i).zfill(4)}.png")
                out_sampledImg_path = os.path.join(logger.get_dir(), "sampledImg", f"{str(count + i).zfill(4)}.png")
                out_outImg_path = os.path.join(logger.get_dir(), "outImg", f"{str(count + i).zfill(4)}.png")
                out_coarseImg_path = os.path.join(logger.get_dir(), "coarseImg", f"{str(count + i).zfill(4)}.png")
                
                tmp_ones = th.ones(gt[i].shape) * (-1)
                inputImg = gt[i].to(mask.device) * mask[i] + (1 - mask[i]) * tmp_ones.to(mask.device)
                sampledImg = sample_fine[i].unsqueeze(0)
                outImg = mask[i] * inputImg + (1 - mask[i]) * sampledImg
                
                if args.use_cf:
                    coarseImg = sample_coarse_256[i].unsqueeze(0)
                    out_coarseImg = mask[i] * inputImg + (1 - mask[i]) * coarseImg
                
                gtImg = gt[i]
                gtImg = gtImg.reshape(outImg.shape).to(outImg.device)
                
                utils.save_image(gtImg, out_gtImg_path, nrow=1, normalize=True, value_range=(-1, 1))
                utils.save_image(inputImg, out_inputImg_path, nrow=1, normalize=True, value_range=(-1, 1))
                utils.save_image(sampledImg, out_sampledImg_path, nrow=1, normalize=True, value_range=(-1, 1))
                utils.save_image(outImg, out_outImg_path, nrow=1, normalize=True, value_range=(-1, 1))
                if args.use_cf:
                    utils.save_image(out_coarseImg, out_coarseImg_path, nrow=1, normalize=True, value_range=(-1, 1))
                
                lpips_value += calculate_lpips(gtImg, outImg)
                psnr_value += calculate_psnr(gtImg, outImg)
                ssim_value += calculate_ssim(gtImg, outImg)
                l1_value += calculate_l1(gtImg, outImg)
                if args.use_cf:
                    coarse_lpips_value += calculate_lpips(gtImg, out_coarseImg)
                    coarse_psnr_value += calculate_psnr(gtImg, out_coarseImg)
                    coarse_ssim_value += calculate_ssim(gtImg, out_coarseImg)
                    coarse_l1_value += calculate_l1(gtImg, out_coarseImg)
            
            count += args.batch_size
            
            with open(metrics_file_path, "a") as metrics_file:
                if args.use_cf:
                    metrics_file.write(f"Coarse {count} samples LPIPS: {coarse_lpips_value / count:.4f}\n")
                    metrics_file.write(f"Coarse {count} samples PSNR: {coarse_psnr_value / count:.4f}\n")
                    metrics_file.write(f"Coarse {count} samples SSIM: {coarse_ssim_value / count:.4f}\n")
                    metrics_file.write(f"Coarse {count} samples L1(%): {coarse_l1_value / count * 100:.2f}\n")
                metrics_file.write(f"{count} samples LPIPS: {lpips_value / count:.4f}\n")
                metrics_file.write(f"{count} samples PSNR: {psnr_value / count:.4f}\n")
                metrics_file.write(f"{count} samples SSIM: {ssim_value / count:.4f}\n")
                metrics_file.write(f"{count} samples L1(%): {l1_value / count * 100:.2f}\n")
                metrics_file.write(f"\n")
           
            logger.log(f"created {count} samples")
        except Exception as e:
            logger.log(f"Error in sampling loop: {str(e)}")
            raise
    
    dist.barrier()
    logger.log("sampling complete")
    end_time = time.time()
    total_time = end_time - start_time
    each_time = total_time / count
    logger.log(f"Total time: {total_time}.")
    logger.log(f"Each time: {each_time}.")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=5,
        base_samples="",
        model_path_64="",
        model_path_256="",
        save_dir="",
        mask_path="",
        data_dir="",
        t_T=250,
        n_sample=1,
        inpa_inj_sched_prev=True,
        inpa_inj_sched_prev_cumnoise=False,
        use_cf=False,
        use_inverse_masks=False,
        special_mask=False,
        t_T_fine=50,
        dpm_steps=20,
        dpm_steps_coarse=20,
        dpm_steps_fine=20,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
