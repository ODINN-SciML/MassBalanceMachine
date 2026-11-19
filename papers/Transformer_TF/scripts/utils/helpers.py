import torch
import numpy as np
import random
import gc
import os
import shutil


def seed_all(seed=None):
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Forbid nondeterministic ops (warn if an op has no deterministic impl)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Setting CUBLAS environment variable (helps in newer versions)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def free_up_cuda():
    """Frees up unused CUDA memory in PyTorch."""
    gc.collect()  # Run garbage collection
    torch.cuda.empty_cache()  # Free unused cached memory
    torch.cuda.ipc_collect()  # Collect inter-process memory


def emptyfolder(path):
    """Removes all files and subdirectories in the given folder."""
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Remove file
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove folder and all contents
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
    else:
        os.makedirs(path, exist_ok=True)  # Ensure directory exists
