import subprocess
import sys

def install(package_args):
    """Run a pip install command with support for extra arguments."""
    # Base command
    command = [sys.executable, "-m", "pip", "install"]
    # Split package_args into a list if it contains spaces (e.g., for --extra-index-url)
    args = package_args.split()
    # Extend command with all arguments
    command.extend(args)
    try:
        subprocess.check_call(command)
        print(f"Successfully installed: {' '.join(args)}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {' '.join(args)}: {e}")
        sys.exit(1)
# Install PyTorch and its ecosystem together to enforce consistency
#install("torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116")
install("torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121")
# Verify installation
# Install other packages (avoid upgrading torch)
packages = [
    "numpy==1.24.4",
    "pillow",
    "lpips",
    "torchmetrics==0.11.4",  # Pin to a version compatible with PyTorch 1.13.1
        "click",
    "scipy",
    "psutil",
    "requests",
    "tqdm",
    "blobfile",
    "imageio",
    "imageio-ffmpeg",
    "pyspng",
    "omegaconf",
    "pytorch_lightning==1.9.5",  # Pin to a version compatible with PyTorch 1.13.1
    "einops",
    "taming-transformers",
    "transformers"
]

for package in packages:
    install(package)

# Verify key installations
print("Verifying installations...")
try:
    import torch
    import numpy as np
    import torchvision
    import torchaudio
    import torchmetrics
    import pytorch_lightning
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"NCCL available: {torch.distributed.is_nccl_available()}")
    print(f"NumPy version: {np.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    print(f"Torchmetrics version: {torchmetrics.__version__}")
    print(f"Pytorch Lightning version: {pytorch_lightning.__version__}")
except ImportError as e:
    print(f"Verification failed: {e}")
    sys.exit(1)