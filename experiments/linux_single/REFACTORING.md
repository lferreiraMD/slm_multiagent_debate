# Refactoring Summary: config.yaml as Single Source of Truth

**Date:** 2025-11-28
**Objective:** Eliminate hardcoded model specifications and use `config.yaml` as the single repository for all vLLM model definitions.

## Changes Made

### 1. config.yaml Enhancement

**Added `model_metadata` section** (lines 76-132):

```yaml
model_metadata:
  vllm-qwen3-0.6b:
    vram_gb: 2
    params: "0.6B"
    description: "Smallest Qwen3 model"

  # ... (11 total vLLM models with metadata)
```

**Purpose:**
- Centralized VRAM requirements for all vLLM models
- Parameter counts and descriptions
- Used for automatic filtering based on hardware constraints

### 2. generate_job_configs.py Refactoring

**Before:**
- Hardcoded `MODELS` list (9 models)
- Hardcoded VRAM comments
- Static configuration

**After:**
- Dynamic model loading from config.yaml via `load_vllm_models()`
- CLI argument `--max-vram-gb` for flexible VRAM limits
- Automatic filtering and exclusion of models exceeding VRAM limit
- Clear output showing included/excluded models

**New Features:**
```bash
# Default 24GB limit (single RTX 3090)
python3 generate_job_configs.py

# Custom VRAM limit (16GB)
python3 generate_job_configs.py --max-vram-gb 16

# Dual GPU setup (48GB)
python3 generate_job_configs.py --max-vram-gb 48
```

**Key Functions:**
- `load_vllm_models(max_vram_gb)`: Reads config.yaml, filters models, returns sorted list
- `generate_task_jobs(task_name, task_config, models)`: Now accepts models parameter
- `main()`: Parses CLI args, loads models dynamically, generates configs

### 3. Documentation Updates

#### experiments/linux_single/README.md
- Added "Model Selection (config.yaml)" section
- Documented how to add new models
- Explained VRAM limit customization
- Updated "Excluded Models" section to reference dynamic filtering

#### CLAUDE.md
- Added `model_metadata` to Configuration section
- Updated Known Issues to mention dynamic filtering
- Clarified that config.yaml is single source of truth

## Benefits

### 1. Maintainability
- **Single source:** All model specs in one place (config.yaml)
- **No duplication:** No need to update multiple files when adding models
- **Version control:** Model metadata tracked in git

### 2. Flexibility
- **Hardware-aware:** Automatically adapts to different GPU VRAM limits
- **Expandable:** Easy to add new models (just edit config.yaml)
- **Reusable:** Same config.yaml can be used by experiments/linux/ and experiments/linux_single/

### 3. User Experience
- **Clear output:** Shows exactly which models are included/excluded and why
- **CLI control:** Easy VRAM limit override via `--max-vram-gb`
- **Self-documenting:** VRAM requirements visible in config.yaml

## Validation Results

### Test 1: Default 24GB VRAM (Single RTX 3090)
```
Valid models: 9
  ✓ vllm-qwen3-0.6b    (0.6B, ~2GB VRAM)
  ✓ vllm-vibethinker   (1.5B, ~4GB VRAM)
  ✓ vllm-deepseek      (1.5B, ~4GB VRAM)
  ✓ vllm-qwen3-1.7b    (1.7B, ~4GB VRAM)
  ✓ vllm-llama32-3b    (3B, ~7GB VRAM)
  ✓ vllm-smallthinker  (3B, ~7GB VRAM)
  ✓ vllm-qwen3-4b      (4B, ~9GB VRAM)
  ✓ vllm-qwen3-8b      (8B, ~16GB VRAM)
  ✓ vllm-llama31-8b    (8B, ~17GB VRAM)

Excluded models: 2
  ✗ vllm-qwen3-14b     (14B, ~28GB VRAM - exceeds 24GB limit)
  ✗ vllm-oss-20b       (20B, ~40GB VRAM - exceeds 24GB limit)

Total jobs: 216 (9 models × 6 agent counts × 4 tasks)
```

### Test 2: Custom 16GB VRAM
```
Valid models: 8 (excludes vllm-llama31-8b)
Excluded models: 3
Total jobs: 192 (8 models × 6 agent counts × 4 tasks)
```

### Test 3: Config File Verification
```
$ wc -l configs/persona_*.txt
    55 configs/persona_biography_jobs.txt
    55 configs/persona_gsm_jobs.txt
    55 configs/persona_math_jobs.txt
    55 configs/persona_mmlu_jobs.txt
   220 total

$ head -2 configs/persona_math_jobs.txt
job_id,model_alias,n_agents,rounds,task,num_param,num_value,random_seed,personas_tuple
1,vllm-qwen3-0.6b,2,3,math,num_problems,100,0,"('...')"
```

## Migration Guide

### Adding a New vLLM Model

1. **Edit config.yaml** (add to both sections):

```yaml
models:
  vllm-new-model: "org/new-model-name"

model_metadata:
  vllm-new-model:
    vram_gb: 12
    params: "5B"
    description: "New 5B parameter model"
```

2. **Regenerate configs:**

```bash
cd experiments/linux_single
python3 generate_job_configs.py
```

3. **Verify:**

```bash
# Check model appears in output
python3 generate_job_configs.py | grep vllm-new-model
```

### Adjusting for Different Hardware

```bash
# RTX 4090 (24GB, same as 3090)
python3 generate_job_configs.py --max-vram-gb 24

# RTX 3060 (12GB)
python3 generate_job_configs.py --max-vram-gb 12

# A100 (40GB)
python3 generate_job_configs.py --max-vram-gb 40

# Dual RTX 3090 (48GB total)
python3 generate_job_configs.py --max-vram-gb 48
```

## Files Modified

### Created
- `config.yaml` → Added `model_metadata` section (56 lines)

### Refactored
- `experiments/linux_single/generate_job_configs.py` → Complete rewrite (245 lines)
  - Added `load_vllm_models()` function
  - Added CLI argument parsing
  - Removed hardcoded MODELS list
  - Dynamic model loading and filtering

### Updated Documentation
- `experiments/linux_single/README.md` → Added "Model Selection (config.yaml)" section
- `CLAUDE.md` → Updated Configuration and Known Issues sections

### Unchanged (Shell Scripts)
- `run_persona_math.sh`, `run_persona_gsm.sh`, `run_persona_biography.sh`, `run_persona_mmlu.sh`
- **Note:** These still use heuristic-based VRAM estimation in `check_gpu_memory()` function
- **Rationale:** Simpler than calling Python for each check; model filtering already handled by config generation

## Backward Compatibility

✅ **Fully backward compatible** for existing workflows:

```bash
# Old way still works (uses default 24GB)
bash setup.sh

# New way also works
python3 generate_job_configs.py --max-vram-gb 24
```

## Future Enhancements

1. **Extend to experiments/linux/** - Apply same pattern to dual-GPU setup
2. **Runtime VRAM detection** - Auto-detect GPU VRAM and set `--max-vram-gb`
3. **Model download integration** - Use model_metadata in download scripts
4. **Shell script integration** - Make `check_gpu_memory()` read from config.yaml instead of heuristics

## Conclusion

✅ **Design Goal Achieved:** config.yaml is now the **single source of truth** for all vLLM model specifications.

**Impact:**
- Easier maintenance (one place to update)
- Better flexibility (hardware-aware filtering)
- Improved user experience (clear output, CLI control)
- Future-proof (easy to extend to other experiments)
