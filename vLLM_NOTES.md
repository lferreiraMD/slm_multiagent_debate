# vLLM API Notes: Multi-GPU Optimization & Best Practices

**Last Updated:** November 26, 2025
**vLLM Version:** 0.11.0
**Hardware Context:** 2x RTX 3090 (24GB each = 48GB total VRAM)

---

## Table of Contents
1. [Multi-GPU Strategies](#multi-gpu-strategies)
2. [Critical Parameters](#critical-parameters)
3. [Optimal Configurations](#optimal-configurations)
4. [Memory Management & Cleanup](#memory-management--cleanup)
5. [Performance Optimization](#performance-optimization)
6. [Common Pitfalls](#common-pitfalls)

---

## Multi-GPU Strategies

### 1. Tensor Parallelism (TP) ⭐ **Best for 2x GPUs in Same Node**

**How it works:**
- Splits model weights across GPUs horizontally
- Each GPU processes the same batch but different parts of the model
- Requires high-bandwidth interconnect (NVLink/InfiniBansd)

**Benefits:**
- Reduces per-GPU memory usage
- **Improves latency** (compute parallelized)
- Super-linear scaling: More KV cache → larger batches → better throughput
- Memory bandwidth effectively multiplied

**Requirements:**
- `n_q_heads / tensor_parallel_size = whole number`
- High-bandwidth interconnect (NVLink between RTX 3090s = ✅ Good!)

**Configuration:**
```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,  # Use both GPUs
    gpu_memory_utilization=0.9,  # Can be aggressive with TP
    ...
)
```

**When to use:**
- **Your case:** 2x RTX 3090 in same node with NVLink
- Models that don't fit on single GPU
- Latency-sensitive applications

---

### 2. Pipeline Parallelism (PP)

**How it works:**
- Splits model layers across GPUs vertically
- GPU 0: Layers 1-16, GPU 1: Layers 17-32
- Sequential processing (pipeline stages)

**Benefits:**
- Reduces memory per GPU
- Works with slower interconnects

**Drawbacks:**
- **Does NOT improve latency** (sequential bottleneck)
- Pipeline bubbles (idle time)

**Configuration:**
```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    pipeline_parallel_size=2,  # Use both GPUs
    ...
)
```

**When to use:**
- Multi-node setups with slow interconnects
- Very large models that don't fit even with TP
- **NOT recommended for your 2x RTX 3090 setup** (use TP instead)

---

### 3. Data Parallelism (DP)

**How it works:**
- Each GPU loads full model
- Different GPUs process different requests
- No inter-GPU communication during inference

**Benefits:**
- Maximum throughput for many concurrent requests
- No synchronization overhead

**Configuration:**
```python
# Automatic with vLLM's engine (no parameter needed)
# Just run multiple LLM instances or use OpenAI server
```

**When to use:**
- High-throughput serving with many concurrent requests
- Model fits comfortably on single GPU

---

### 4. Hybrid: TP + DP (Advanced)

**Configuration:**
```bash
# 4 GPUs total: 2 TP replicas, each using 2 GPUs
vllm serve model \
  --tensor-parallel-size 2 \
  --num-replicas 2
```

**When to use:**
- 4+ GPUs available
- Need both low latency (TP) and high throughput (DP)

---

## Critical Parameters

### `tensor_parallel_size` ⭐ **MOST IMPORTANT FOR YOU**

**Purpose:** Distribute model across GPUs within a node

**Your optimal value:** `2` (use both RTX 3090s)

**Rules:**
- ✅ **ALWAYS set to number of GPUs** (never use `tensor_parallel_size=1` on multi-GPU!)
- Must satisfy: `n_q_heads % tensor_parallel_size == 0`
- Best practice: Set to GPUs per node

**Performance impact:**
- `tensor_parallel_size=1` on 2 GPUs: **Severe performance degradation** ❌
- `tensor_parallel_size=2` on 2 GPUs: **Optimal performance** ✅

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,  # ← Critical for your setup
)
```

---

### `gpu_memory_utilization`

**Purpose:** Fraction of GPU memory reserved for model + KV cache

**Default:** `0.9` (90%)

**Recommended range:** `0.5 - 0.95`

**Guidelines:**
- **0.5-0.7:** Safe, prevents OOM, good for sequential model loading
- **0.8-0.9:** Aggressive, maximizes KV cache, better throughput
- **0.95:** Production use with known workload

**Your use cases:**
```python
# Model downloading (sequential loads)
gpu_memory_utilization=0.7  # Leave headroom for cleanup

# Production inference (single model)
gpu_memory_utilization=0.9  # Maximize throughput

# Multiagent debate (multiple models)
gpu_memory_utilization=0.8  # Balance between performance and stability
```

---

### `max_model_len`

**Purpose:** Limit context window to reduce KV cache memory

**Impact:** Halving `max_model_len` doubles potential concurrency

**Guidelines:**
```python
# Large context needed (biography task)
max_model_len=8192

# Short context sufficient (math, GSM)
max_model_len=2048  # Saves significant memory

# Debugging/testing
max_model_len=512  # Minimal memory
```

**Memory savings:**
- 131K → 2K tokens: ~16GB → ~0.25GB KV cache (98% reduction!)

---

### `enable_chunked_prefill` 🚀 **New Performance Feature**

**Purpose:** Process large prompts in chunks, batch with decode requests

**Benefits:**
- Better latency (fewer prefills blocking decodes)
- Higher throughput (better GPU utilization)
- Balances compute-bound (prefill) and memory-bound (decode) ops

**Configuration:**
```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,  # Tune for your workload
)
```

**Tuning:**
- Smaller `max_num_batched_tokens` (e.g., 2048): Better inter-token latency
- Larger `max_num_batched_tokens` (e.g., 8192+): Better time-to-first-token
- Recommendation: Start with 8192 for smaller models on large GPUs

---

### `enforce_eager`

**Purpose:** Disable CUDA graph compilation

**Impact:**
- ✅ Reduces memory usage (~1-2GB saved)
- ❌ Slower inference (no graph optimization)
- ✅ Fixes flash-attn compatibility issues

**Use cases:**
```python
# Model downloading/caching (don't care about speed)
enforce_eager=True

# Production inference (care about latency)
enforce_eager=False  # Default
```

---

### `enable_sleep_mode` ⭐ **Critical for Cleanup**

**Purpose:** Enable `sleep()` method for GPU memory offloading

**Configuration:**
```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_sleep_mode=True,  # ← Required for proper cleanup
)

# After inference
llm.sleep(level=1)  # Offload weights to CPU, clear KV cache
del llm             # Now cleanup works properly
```

**Sleep levels:**
- `level=1`: Offload weights to CPU, clear KV cache (recommended)
- `level=2`: Unload weights AND KV cache entirely

**Why this matters:**
- vLLM has **known bug**: `del llm` doesn't free GPU memory properly
- Without `sleep()`, you get **stuck EngineCore processes**
- This is the **official workaround** until bug is fixed

---

## Optimal Configurations

### Your Hardware: 2x RTX 3090 (48GB Total)

#### Configuration 1: Single Large Model (8-14B)
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,           # ⭐ Use both GPUs
    gpu_memory_utilization=0.9,       # Maximize KV cache
    max_model_len=8192,               # Full context
    enable_chunked_prefill=True,      # Better throughput
    max_num_batched_tokens=8192,
    enable_sleep_mode=True,           # Proper cleanup
)

# Use model...
outputs = llm.generate(prompts)

# Cleanup
llm.sleep(level=1)
del llm
```

**Expected performance:**
- 8B model: ~100 tokens/sec
- 14B model: ~50-70 tokens/sec
- Can handle 8K context with good batch sizes

---

#### Configuration 2: Sequential Model Loading (Your Download Script)
```python
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,           # Single GPU (sequential loads)
    gpu_memory_utilization=0.7,       # Conservative (room for cleanup)
    max_model_len=2048,               # Minimal KV cache
    enforce_eager=True,               # Save memory, fix flash-attn
    disable_custom_all_reduce=True,   # Avoid kernel issues
    enable_sleep_mode=True,           # ⭐ Critical for cleanup
)

# Download/test model...

# Cleanup before next model
llm.sleep(level=1)                    # ⭐ Offload to CPU first
del llm

# Aggressive cleanup
from utils import release_cuda_memory
release_cuda_memory(delay=2.0, verbose=True)
```

**Why these settings:**
- `tensor_parallel_size=1`: Load one model at a time on GPU 0
- `gpu_memory_utilization=0.7`: Leave 30% headroom for next model
- `enforce_eager=True`: Bypass flash-attn bugs, save memory
- `enable_sleep_mode=True`: **Must have** for proper cleanup

---

#### Configuration 3: Multiagent Debate (3-5 Small Models)
```python
# Agent 1
llm1 = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    tensor_parallel_size=1,           # Each on separate GPU
    gpu_memory_utilization=0.8,       # Moderate
    max_model_len=4096,
)

# Agent 2 (different GPU)
llm2 = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
)

# Distribute agents across GPUs manually or use Ray
```

**Alternative:** Use `tensor_parallel_size=2` for larger models, run agents sequentially

---

## Memory Management & Cleanup

### The Problem: vLLM Cleanup Bug

**Known issues (as of v0.11.0):**
1. LLM class has **no `__del__` method**
2. `del llm` does **NOT free GPU memory** reliably
3. Worker processes (EngineCore) become **zombies**
4. GitHub issues: #1908, #5211, #5104, #16667

**Symptoms:**
```bash
$ nvidia-smi
# Shows: VLLM::EngineCore using 16GB (even after script exits)
```

---

### The Solution: sleep() + Manual Cleanup

**Step 1: Enable sleep mode**
```python
llm = LLM(model="...", enable_sleep_mode=True)
```

**Step 2: Sleep before deletion**
```python
llm.sleep(level=1)  # Offload weights, clear KV cache
del llm
```

**Step 3: Manual cleanup**
```python
from utils import release_cuda_memory
release_cuda_memory(delay=2.0, verbose=True)
```

**What `release_cuda_memory()` does:**
1. Destroys distributed process groups
2. Synchronizes CUDA operations
3. Clears CUDA cache + IPC memory
4. Forces garbage collection
5. Waits 2 seconds for OS cleanup

---

### Complete Cleanup Example

```python
from vllm import LLM
from utils import release_cuda_memory
import torch

# Create model
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_sleep_mode=True,  # ← Required
)

# Use model
outputs = llm.generate(prompts)

# Cleanup sequence
print("Cleaning up...")
llm.sleep(level=1)           # 1. Offload to CPU
del llm                      # 2. Delete Python object
release_cuda_memory(         # 3. Manual cleanup
    delay=2.0,
    verbose=True
)

# Verify cleanup
torch.cuda.empty_cache()
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

### Monitoring & Debugging

**Check GPU processes:**
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**Kill stuck processes:**
```bash
# Find stuck vLLM processes
nvidia-smi --query-compute-apps=pid,process_name --format=csv

# Kill them
kill -9 <PID>
```

**Use the test utility:**
```bash
python3 utils/test_cuda_cleanup.py
```

This shows:
- GPU memory before/after cleanup
- Active GPU processes
- Stuck vLLM EngineCore processes
- Cleanup effectiveness

---

## Performance Optimization

### 1. Always Use Tensor Parallelism on Multi-GPU

**❌ Wrong:**
```python
# On 2x RTX 3090 machine
llm = LLM(model="...", tensor_parallel_size=1)  # ❌ SEVERE performance hit!
```

**✅ Correct:**
```python
# On 2x RTX 3090 machine
llm = LLM(model="...", tensor_parallel_size=2)  # ✅ Use both GPUs!
```

**Performance difference:** Up to 10x slower with `tensor_parallel_size=1`!

---

### 2. Tune Memory for Throughput

**Higher memory utilization → More KV cache → Larger batches → Better throughput**

```python
# Conservative (debugging, testing)
gpu_memory_utilization=0.6  # ~60 tokens/sec

# Balanced (general use)
gpu_memory_utilization=0.8  # ~80 tokens/sec

# Aggressive (production)
gpu_memory_utilization=0.9  # ~100 tokens/sec
```

**Super-linear scaling:** Doubling KV cache can more than double throughput due to:
- Larger batch sizes
- Better memory locality
- Reduced scheduling overhead

---

### 3. Enable Chunked Prefill

```python
llm = LLM(
    model="...",
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,  # Start here, tune for workload
)
```

**Benefits:**
- 10-20% throughput improvement
- Better latency for mixed workloads
- Especially helpful for long prompts

---

### 4. Right-Size Context Window

**Don't use more context than you need:**

```python
# Biography task: needs 8K context
max_model_len=8192

# Math/GSM: only needs 2K context
max_model_len=2048  # ← 75% memory savings!
```

**Rule of thumb:** Context memory scales linearly with `max_model_len`

---

### 5. Choose Right Parallelism Strategy

**Your 2x RTX 3090 Setup:**

| Model Size | Strategy | Configuration |
|------------|----------|---------------|
| ≤3B | Single GPU | `tensor_parallel_size=1` |
| 3-8B | Tensor Parallel | `tensor_parallel_size=2` ⭐ |
| 8-14B | Tensor Parallel | `tensor_parallel_size=2` ⭐ |
| >14B | May not fit | Consider quantization |

---

## Common Pitfalls

### ❌ Pitfall 1: Using `tensor_parallel_size=1` on Multi-GPU

**Problem:**
```python
# On 2x GPU machine
llm = LLM(model="...", tensor_parallel_size=1)  # ❌
```

**Why it's bad:**
- vLLM's distributed architecture expects parallelism
- Causes severe performance degradation (up to 10x slower)
- Wastes second GPU

**Fix:**
```python
llm = LLM(model="...", tensor_parallel_size=2)  # ✅
```

---

### ❌ Pitfall 2: Not Enabling Sleep Mode

**Problem:**
```python
llm = LLM(model="...")  # No enable_sleep_mode
# ...
del llm  # ❌ Memory not freed, EngineCore stays alive
```

**Fix:**
```python
llm = LLM(model="...", enable_sleep_mode=True)
# ...
llm.sleep(level=1)  # ← Must call before delete
del llm
```

---

### ❌ Pitfall 3: Too High Memory Utilization for Sequential Loads

**Problem:**
```python
# Loading models sequentially
for model_path in models:
    llm = LLM(model_path, gpu_memory_utilization=0.9)  # ❌ OOM!
    del llm
```

**Why:** Not enough headroom for next model during cleanup

**Fix:**
```python
for model_path in models:
    llm = LLM(
        model_path,
        gpu_memory_utilization=0.7,  # ✅ Conservative
        enable_sleep_mode=True
    )
    llm.sleep(level=1)
    del llm
    release_cuda_memory(delay=2.0)  # Extra safety
```

---

### ❌ Pitfall 4: Using Pipeline Parallelism Within Node

**Problem:**
```python
# On 2x RTX 3090 (same node with NVLink)
llm = LLM(
    model="...",
    pipeline_parallel_size=2,  # ❌ Wrong choice
)
```

**Why it's bad:**
- Pipeline parallel doesn't improve latency
- Introduces pipeline bubbles (idle time)
- Wastes NVLink bandwidth

**Fix:**
```python
# On same node with NVLink
llm = LLM(
    model="...",
    tensor_parallel_size=2,  # ✅ Use TP instead
)
```

**Rule:** TP within node, PP across nodes (if needed)

---

### ❌ Pitfall 5: Incompatible Head Count

**Problem:**
```python
# Model has 32 attention heads
llm = LLM(
    model="...",
    tensor_parallel_size=5,  # ❌ 32 % 5 != 0
)
```

**Error:** `n_q_heads must be divisible by tensor_parallel_size`

**Fix:** Choose TP size that divides head count evenly:
- 32 heads: Use TP size 1, 2, 4, 8, 16, 32 ✅
- 32 heads: Don't use TP size 3, 5, 6, 7 ❌

---

## Quick Reference

### For Your 2x RTX 3090 Setup

**Production Inference (Single Model):**
```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,
    enable_sleep_mode=True,
)
```

**Model Downloads (Sequential Loading):**
```python
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_model_len=2048,
    enforce_eager=True,
    enable_sleep_mode=True,
)
llm.sleep(level=1)
del llm
release_cuda_memory(delay=2.0)
```

**Multiagent Debate (3-5 Agents):**
```python
# Option 1: Sequential (safe)
for agent_model in agent_models:
    llm = LLM(model=agent_model, tensor_parallel_size=1)
    # Use agent...
    llm.sleep(level=1)
    del llm

# Option 2: Parallel (if models fit)
llms = [
    LLM(model=m, tensor_parallel_size=1, gpu_memory_utilization=0.6)
    for m in agent_models[:2]  # 2 agents max simultaneously
]
```

---

## Automatic GPU Detection & Configuration

### Overview

The codebase now includes **automatic GPU detection and optimal parameter configuration** for vLLM. This feature:
- ✅ Detects number of GPUs, VRAM, and NVLink availability
- ✅ Automatically sets optimal `tensor_parallel_size`, `gpu_memory_utilization`, etc.
- ✅ Adapts to different hardware configurations (1-GPU, 2-GPU, N-GPU)
- ✅ Prevents common pitfalls (e.g., `tensor_parallel_size=1` on multi-GPU)
- ✅ Works only with vLLM backend (doesn't affect MLX or Ollama)

---

### How It Works

**Module:** `utils/gpu_config.py`

**Key Functions:**
```python
from utils import (
    is_vllm_backend,          # Check if vLLM is available
    detect_vllm_gpus,         # Detect GPU configuration
    get_vllm_optimal_config,  # Get optimal vLLM parameters
    print_gpu_summary,        # Print GPU info
    get_gpu_info_string       # Get concise GPU string
)
```

**Detection:**
```python
gpu_info = detect_vllm_gpus()
# Returns:
# {
#     'count': 2,
#     'models': ['RTX 3090', 'RTX 3090'],
#     'vram_per_gpu_gb': 24.0,
#     'total_vram_gb': 48.0,
#     'has_nvlink': True,
#     'available_vram_gb': 45.2,
# }
```

**Configuration:**
```python
config = get_vllm_optimal_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    use_case='production',  # or 'download' or 'debate'
    gpu_info=gpu_info       # Optional, auto-detected if None
)

llm = LLM(model=model_name, **config)
```

---

### Use Cases

#### 1. Download (Sequential Loading)
```python
config = get_vllm_optimal_config(model_name, use_case='download')
# Returns:
# {
#     'tensor_parallel_size': 1,
#     'gpu_memory_utilization': 0.7,
#     'max_model_len': 2048,
#     'enforce_eager': True,
#     'enable_sleep_mode': True,
#     'disable_custom_all_reduce': True,
# }
```

#### 2. Production (Maximum Throughput)
```python
config = get_vllm_optimal_config(model_name, use_case='production')
# On 2x GPU system:
# {
#     'tensor_parallel_size': 2,  # ⭐ Uses both GPUs
#     'gpu_memory_utilization': 0.9,
#     'max_model_len': 8192,
#     'enable_chunked_prefill': True,
#     'max_num_batched_tokens': 8192,
#     'enable_sleep_mode': True,
#     'disable_custom_all_reduce': True,
# }
```

#### 3. Debate (Multiple Models)
```python
config = get_vllm_optimal_config(model_name, use_case='debate')
# Returns:
# {
#     'tensor_parallel_size': 1,  # Each agent separate
#     'gpu_memory_utilization': 0.8,
#     'max_model_len': 4096,
#     'enable_sleep_mode': True,
#     'disable_custom_all_reduce': True,
# }
```

---

### Example: download_models.py

The model download script now uses auto-configuration by default:

```bash
# Auto-configure based on detected GPUs
python3 experiments/download_models.py

# Disable auto-configuration
python3 experiments/download_models.py --disable-auto-gpu
```

**Output:**
```
vLLM Backend - GPU Configuration
======================================================================
  GPUs: 2x RTX 3090
  VRAM per GPU: 24.0 GB
  Total VRAM: 48.0 GB
  Available VRAM: 45.2 GB
  NVLink: ✓

Auto-Configuration (use_case='download'):
  disable_custom_all_reduce: True
  enable_sleep_mode: True
  enforce_eager: True
  gpu_memory_utilization: 0.7
  max_model_len: 2048
  tensor_parallel_size: 1
======================================================================
```

---

### How to Use in Your Code

**Option 1: Automatic (Recommended)**
```python
from vllm import LLM
from utils import detect_vllm_gpus, get_vllm_optimal_config

# Auto-detect and configure
gpu_info = detect_vllm_gpus()
config = get_vllm_optimal_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    use_case='production',
    gpu_info=gpu_info
)

llm = LLM(model=model_name, **config)
```

**Option 2: With Overrides**
```python
config = get_vllm_optimal_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    use_case='production'
)

# Override specific parameters
config['max_model_len'] = 4096  # Custom context length

llm = LLM(model=model_name, **config)
```

**Option 3: Manual (When vLLM Not Available)**
```python
from utils import is_vllm_backend

if is_vllm_backend():
    config = get_vllm_optimal_config(model_name, use_case='production')
else:
    # MLX or Ollama backend - use their own configuration
    config = {}

llm = LLM(model=model_name, **config)
```

---

### Benefits

✅ **Portable:** Works across different HPC systems without code changes
✅ **Optimal:** Automatically chooses best `tensor_parallel_size` for your GPU count
✅ **Safe:** Prevents common mistakes (e.g., wrong tensor_parallel_size)
✅ **Flexible:** Easy to override specific parameters
✅ **Backend-specific:** Only applies to vLLM, doesn't interfere with MLX/Ollama

---

### Technical Details

**GPU Detection:**
- Uses `torch.cuda.device_count()` for GPU count
- Uses `torch.cuda.get_device_properties()` for VRAM
- Uses `nvidia-smi topo -m` for NVLink detection
- Estimates model size from name (e.g., "7B" → ~14GB)

**Configuration Logic:**
- **Single GPU:** `tensor_parallel_size=1`, conservative memory
- **Multi-GPU + Model fits:** `tensor_parallel_size=N` (per vLLM best practices)
- **Multi-GPU + Model doesn't fit:** `tensor_parallel_size=N` (required)
- **NVLink warning:** Warns if using TP without NVLink

**Use Case Tuning:**
- **Download:** Conservative, prevents OOM during sequential loads
- **Production:** Aggressive, maximizes throughput
- **Debate:** Balanced, allows multiple simultaneous models

---

## Resources

- **Official Docs:** https://docs.vllm.ai/en/stable/
- **GitHub Issues (Cleanup):** #1908, #5211, #5104, #16667
- **Multi-GPU Guide:** https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
- **Optimization Guide:** https://docs.vllm.ai/en/latest/configuration/optimization/

---

## Notes for Future Investigation

1. **Quantization:** Test FP8/INT8 quantization for larger models
2. **Prefix Caching:** Explore for multiagent debate (shared prompts)
3. **Ray Integration:** For multi-node scaling beyond 2 GPUs
4. **Model Profiling:** Benchmark throughput/latency for each model size
5. **Custom Attention Backend:** Test alternatives to flash-attn if issues persist
