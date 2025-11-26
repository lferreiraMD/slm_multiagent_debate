"""
CUDA memory cleanup utilities for vLLM and PyTorch.

This module provides functions to aggressively release GPU memory,
especially useful when loading models sequentially or cleaning up
after vLLM LLM instances.
"""

import gc
import time
from typing import Optional

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    dist = None


def release_cuda_memory(delay: float = 0.0, verbose: bool = False) -> None:
    """
    Aggressively release CUDA memory and clean up distributed resources.

    This function performs comprehensive cleanup including:
    - Destroying PyTorch distributed process groups
    - Synchronizing CUDA operations
    - Clearing CUDA cache and IPC memory
    - Forcing Python garbage collection
    - Optional delay to ensure complete cleanup

    Args:
        delay: Optional delay in seconds after cleanup (default: 0.0)
               Useful when loading models sequentially to ensure
               memory is fully released before next allocation.
        verbose: Print cleanup status messages (default: False)

    Example:
        >>> from utils.cuda_cleanup import release_cuda_memory
        >>> # After deleting a model
        >>> del model
        >>> release_cuda_memory(delay=2.0, verbose=True)
    """
    if verbose:
        print("  → Releasing CUDA memory...")

    # Destroy distributed process groups (vLLM creates these)
    if TORCH_AVAILABLE and dist is not None:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
                if verbose:
                    print("    ✓ Destroyed process group")
            except Exception as e:
                if verbose:
                    print(f"    ⚠ Could not destroy process group: {e}")

    # CUDA cleanup
    if TORCH_AVAILABLE and torch is not None:
        if torch.cuda.is_available():
            try:
                # Wait for all CUDA operations to complete
                torch.cuda.synchronize()

                # Clear CUDA cache
                torch.cuda.empty_cache()

                # Collect CUDA IPC memory
                torch.cuda.ipc_collect()

                if verbose:
                    print("    ✓ Cleared CUDA cache")
            except Exception as e:
                if verbose:
                    print(f"    ⚠ CUDA cleanup warning: {e}")

    # Force Python garbage collection
    gc.collect()

    if verbose:
        print("    ✓ Garbage collection complete")

    # Optional delay to ensure cleanup completes
    if delay > 0:
        if verbose:
            print(f"    ⏳ Waiting {delay}s for memory release...")
        time.sleep(delay)

    if verbose:
        print("  ✓ Memory cleanup complete")


def get_cuda_memory_stats(device: int = 0) -> Optional[dict]:
    """
    Get current CUDA memory statistics for a device.

    Args:
        device: CUDA device ID (default: 0)

    Returns:
        Dictionary with memory stats in GB, or None if CUDA unavailable:
        - 'allocated': Currently allocated memory
        - 'reserved': Reserved memory by PyTorch
        - 'total': Total GPU memory
        - 'free': Approximate free memory

    Example:
        >>> from utils.cuda_cleanup import get_cuda_memory_stats
        >>> stats = get_cuda_memory_stats()
        >>> print(f"GPU memory: {stats['allocated']:.2f} GB / {stats['total']:.2f} GB")
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None

    try:
        # Convert bytes to GB
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free = total - allocated

        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free
        }
    except Exception as e:
        print(f"Warning: Could not get CUDA memory stats: {e}")
        return None


def print_cuda_memory_summary(device: int = 0) -> None:
    """
    Print a formatted summary of CUDA memory usage.

    Args:
        device: CUDA device ID (default: 0)

    Example:
        >>> from utils.cuda_cleanup import print_cuda_memory_summary
        >>> print_cuda_memory_summary()
        GPU 0 Memory:
          Allocated: 6.24 GB
          Reserved:  6.50 GB
          Total:     23.68 GB
          Free:      17.44 GB
    """
    stats = get_cuda_memory_stats(device)

    if stats is None:
        print("CUDA not available")
        return

    print(f"GPU {device} Memory:")
    print(f"  Allocated: {stats['allocated']:.2f} GB")
    print(f"  Reserved:  {stats['reserved']:.2f} GB")
    print(f"  Total:     {stats['total']:.2f} GB")
    print(f"  Free:      {stats['free']:.2f} GB")
