#!/usr/bin/env python3
"""
Test script for CUDA memory cleanup utilities.

This script demonstrates the cuda_cleanup module by:
1. Checking initial GPU memory with nvidia-smi
2. Running CUDA cleanup
3. Checking final GPU memory and comparing
"""

import subprocess
import sys
from pathlib import Path

# Add parent directory to path to import utils
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from utils.cuda_cleanup import (
    release_cuda_memory,
    get_cuda_memory_stats,
    print_cuda_memory_summary
)


def run_nvidia_smi():
    """Run nvidia-smi and return output."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return None
    except FileNotFoundError:
        print("nvidia-smi not found. Is NVIDIA driver installed?")
        return None


def get_gpu_processes():
    """Get list of processes using GPU memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting GPU processes: {e}")
        return None
    except FileNotFoundError:
        return None


def parse_gpu_processes(output):
    """Parse GPU process list into structured data."""
    processes = []
    if not output or not output.strip():
        return processes

    for line in output.split('\n'):
        if line.strip():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                processes.append({
                    'pid': int(parts[0]),
                    'name': parts[1],
                    'memory_mb': float(parts[2])
                })
    return processes


def print_gpu_processes(processes, label="GPU Processes"):
    """Print formatted list of GPU processes."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")

    if not processes:
        print("  ✓ No processes using GPU memory")
        return

    print(f"\n{'PID':<10} {'Process Name':<30} {'Memory (GB)':>12}")
    print("-" * 70)

    total_memory = 0
    for proc in processes:
        memory_gb = proc['memory_mb'] / 1024
        total_memory += memory_gb
        print(f"{proc['pid']:<10} {proc['name']:<30} {memory_gb:>11.2f} GB")

    print("-" * 70)
    print(f"{'Total':<40} {total_memory:>11.2f} GB")

    # Warn about stuck processes
    stuck_processes = [p for p in processes if 'vllm' in p['name'].lower() or 'EngineCore' in p['name']]
    if stuck_processes:
        print("\n⚠ Warning: Found vLLM processes that may be stuck:")
        for proc in stuck_processes:
            print(f"  - PID {proc['pid']}: {proc['name']} ({proc['memory_mb']/1024:.2f} GB)")
        print("\n  To kill these processes, run:")
        for proc in stuck_processes:
            print(f"    kill -9 {proc['pid']}")


def parse_nvidia_smi_output(output):
    """Parse nvidia-smi output into structured data."""
    gpus = []
    for line in output.split('\n'):
        if line.strip():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'total_mb': float(parts[2]),
                    'used_mb': float(parts[3]),
                    'free_mb': float(parts[4])
                })
    return gpus


def print_gpu_status(gpus, label="GPU Status"):
    """Print formatted GPU status."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    for gpu in gpus:
        total_gb = gpu['total_mb'] / 1024
        used_gb = gpu['used_mb'] / 1024
        free_gb = gpu['free_mb'] / 1024
        usage_pct = (gpu['used_mb'] / gpu['total_mb']) * 100

        print(f"\nGPU {gpu['index']}: {gpu['name']}")
        print(f"  Total:  {total_gb:6.2f} GB")
        print(f"  Used:   {used_gb:6.2f} GB ({usage_pct:5.1f}%)")
        print(f"  Free:   {free_gb:6.2f} GB")


def compare_memory(before, after):
    """Compare memory before and after cleanup."""
    print(f"\n{'='*70}")
    print("Memory Change After Cleanup")
    print(f"{'='*70}")

    for i, (gpu_before, gpu_after) in enumerate(zip(before, after)):
        used_before_gb = gpu_before['used_mb'] / 1024
        used_after_gb = gpu_after['used_mb'] / 1024
        freed_gb = used_before_gb - used_after_gb
        freed_pct = (freed_gb / (gpu_before['total_mb'] / 1024)) * 100

        print(f"\nGPU {i}:")
        print(f"  Before: {used_before_gb:6.2f} GB")
        print(f"  After:  {used_after_gb:6.2f} GB")
        print(f"  Freed:  {freed_gb:6.2f} GB ({freed_pct:5.1f}% of total)")

        if freed_gb > 0.1:  # More than 100 MB freed
            print(f"  ✓ Successfully freed memory")
        elif freed_gb > 0.01:  # Between 10-100 MB
            print(f"  ⚠ Small amount freed (might be normal)")
        else:
            print(f"  ℹ No significant memory freed (might already be clean)")


def main():
    """Main test function."""
    print("="*70)
    print("CUDA Memory Cleanup Test")
    print("="*70)

    # Step 1: Check initial GPU memory with nvidia-smi
    print("\n[Step 1] Checking initial GPU memory with nvidia-smi...")
    nvidia_before = run_nvidia_smi()

    if nvidia_before is None:
        print("Failed to get GPU information. Exiting.")
        return 1

    gpus_before = parse_nvidia_smi_output(nvidia_before)
    print_gpu_status(gpus_before, "Initial GPU Status")

    # Show which processes are using GPU memory
    print("\n" + "="*70)
    print("Checking GPU processes...")
    print("="*70)
    processes_before = parse_gpu_processes(get_gpu_processes())
    print_gpu_processes(processes_before, "Processes Using GPU (Before Cleanup)")

    # Also show PyTorch's view of memory
    print("\n" + "="*70)
    print("PyTorch Memory View (GPU 0)")
    print("="*70)
    print_cuda_memory_summary(device=0)

    # Step 2: Run CUDA cleanup
    print("\n" + "="*70)
    print("[Step 2] Running CUDA cleanup...")
    print("="*70)
    release_cuda_memory(delay=2.0, verbose=True)

    # Step 3: Check final GPU memory
    print("\n[Step 3] Checking final GPU memory...")
    nvidia_after = run_nvidia_smi()

    if nvidia_after is None:
        print("Failed to get GPU information after cleanup.")
        return 1

    gpus_after = parse_nvidia_smi_output(nvidia_after)
    print_gpu_status(gpus_after, "Final GPU Status")

    # Show which processes are still using GPU memory
    print("\n" + "="*70)
    print("Checking GPU processes after cleanup...")
    print("="*70)
    processes_after = parse_gpu_processes(get_gpu_processes())
    print_gpu_processes(processes_after, "Processes Using GPU (After Cleanup)")

    # Show PyTorch's view after cleanup
    print("\n" + "="*70)
    print("PyTorch Memory View After Cleanup (GPU 0)")
    print("="*70)
    print_cuda_memory_summary(device=0)

    # Step 4: Compare before and after
    compare_memory(gpus_before, gpus_after)

    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
