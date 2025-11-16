# ReCogDrive TPU Migration with Torchax

This document outlines the steps and commands to run ReCogDrive training on Google Cloud TPUs using `torchax`.

## Prerequisites

Ensure the following packages are installed on your TPU VM:

```bash
# Install JAX for TPU
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install torchax and optax
pip install git+https://github.com/google/torchax
pip install optax
```

## Migration Summary

1.  **Stage 1 (Vision-Language Pretraining):**
    *   **Script:** `internvl_chat/internvl/train/internvl_chat_finetune_torchax.py`
    *   **Changes:** Replaced Hugging Face `Trainer` and DeepSpeed with a custom training loop using `torchax` and `optax`. Disabled Flash Attention (CUDA-specific) in favor of standard attention which is XLA-compatible.
    *   **Key Features:** Uses `torchax.enable_globally()` to route PyTorch ops to JAX, and `jax.jit` for performance.

2.  **Stage 2 (Diffusion Planner Training):**
    *   **Script:** `navsim/planning/script/run_training_recogdrive_torchax.py`
    *   **Changes:** Replaced PyTorch Lightning with a custom `torchax` training loop. Reused the existing `AgentLightningModule` logic but stripped the Lightning-specific trainer.
    *   **Key Features:** Manages the training loop explicitly with `jax.jit` and `optax` optimizers.

3.  **Codebase Modifications:**
    *   Modified `navsim/agents/recogdrive/recogdrive_agent.py` to remove explicit `.cuda()` calls. The code now dynamically determines the device or relies on input tensor placement, making it compatible with TPU/JAX.

## Lessons Learned & Critical Fixes (Stage 1)

To successfully run Stage 1 training on TPU v6e, the following critical modifications were required:

1.  **JAX/Torchax Compatibility:**
    *   **Tensor Unwrapping:** Before passing `torchax` tensors to `jax.jit` compiled functions (like `train_step`), they must be explicitly unwrapped to JAX arrays using `.jax()` or `torchax.tensor.from_torch(...).jax()`.
    *   **Functional Call:** When using `torch.func.functional_call` inside a JIT-compiled function, input tensors (like `batch`) must be wrapped back into `torchax.tensor.Tensor`. Additionally, passing `batch` as `kwargs` (e.g., `kwargs=torch_batch`) is more robust than positional arguments.
    *   **Loss Unwrapping:** The loss returned by the model (a `torchax` tensor) must be unwrapped to a JAX array (`.jax()`) before being returned by the loss function for `jax.value_and_grad` to work.

2.  **Dynamic Shapes & In-place Operations:**
    *   **Boolean Indexing:** JAX `jit` does not support dynamic boolean indexing (e.g., `tensor[mask]`) because the output shape depends on the data. We replaced this with static shape operations like `cumsum` + `gather` + `where` or simply disabled filtering if applicable (e.g., assuming all images are valid).
    *   **In-place Assignment:** Operations like `tensor[mask] = value` are problematic. We replaced them with `torch.where(mask, value, tensor)`.
    *   **Dynamic Resolution:** To avoid dynamic shapes, we temporarily disabled `dynamic_image_size` and set `max_dynamic_patch` to a fixed value (e.g., 1).

3.  **Memory Management (OOM):**
    *   **FSDP (Sharding):** For large models (e.g., 2B parameters) on TPU, Data Parallelism alone causes OOM due to optimizer states. We implemented **FSDP (Fully Sharded Data Parallel)** using `jax.sharding.Mesh` and `NamedSharding` to shard model parameters and optimizer states across TPU cores.
    *   **BFloat16:** Ensured inputs are cast to `bfloat16` to match model weights and reduce memory usage.
    *   **Gradient Checkpointing:** Enabled `gradient_checkpointing` to trade compute for memory.

4.  **Data Loading:**
    *   **Pin Memory:** Set `pin_memory=False` in `DataLoader` as PyTorch's pinned memory is incompatible with TPU/JAX data transfer.

## Lessons Learned & Critical Fixes (Stage 2)

To successfully run Stage 2 (Diffusion Planner) training on TPU v6e, additional unique challenges were addressed:

1.  **JAX/Torchax Compatibility:**
    *   **Gradient Computation (Params vs Buffers):** `jax.value_and_grad` attempts to compute gradients for *all* arguments passed to the function. However, buffers (like `ddim_t_schedule` which are integers) are not differentiable. **Fix**: Explicitly separated `trainable_params` (passed to `value_and_grad`) and `buffers` (passed as static/auxiliary arguments or just merged inside the loss function).
    *   **In-place Operations:** The diffusion sampler (`ddim`, `ddpm`) relied heavily on in-place PyTorch operations like `clamp_`, `zero_`, `std[0] = ...`. These are invalid in JAX's functional tracing. **Fix**: Replaced all instances with out-of-place equivalents (e.g., `x = x.clamp(...)`, `x = torch.zeros_like(x)`).
    *   **JIT Compilation Errors (Concretization):** `rope.py` contained logic `seq_len = position_ids.max().item() + 1` to dynamically resize the cache. `.item()` forces a concrete value, which fails during JIT tracing where values are abstract. **Fix**: Removed dynamic resizing and assumed a fixed maximum sequence length (sufficient for the planner's short horizon).

2.  **Data Loading & Caching:**
    *   **Directory Structure**: The `CacheOnlyDataset` expects cached tokens to be *directories* containing `.gz` files (e.g., `log/token_id/feature.gz`), not single `.pt` files. Incorrect dummy data generation led to 0 samples being loaded.
    *   **Builder Names**: The dataset loader filters files based on the "unique name" of the builders. We identified and matched the correct names: `internvl_feature` (for VLM features) and `trajectory_target` (for ground truth).
    *   **DataLoader Deadlocks**: PyTorch `DataLoader` with `num_workers > 0` can sometimes deadlock or conflict with TPU runtime in certain Docker environments. **Fix**: Verified functionality with `num_workers=0` and removed incompatible `prefetch_factor` settings.

3.  **Dimension Mismatch:**
    *   The `ReCogDriveDiffusionPlanner` hardcodes the `ego_status_encoder` input dimension to **8**. Our initial dummy data generation used **10**, causing shape mismatches in JAX `einsum` operations. **Fix**: Aligned data generation with the model's expected input dimensions.

## Relationship Between Stage 1 & Stage 2

*   **Stage 1 (Vision-Language Pretraining):**
    *   **Role**: The "Eyes". Fine-tunes the VLM (InternVL) to understand driving scenes from images and text prompts.
    *   **Input**: Camera images, text instructions.
    *   **Output**: A powerful feature extractor that converts visual data into semantic **Hidden States**.
    *   **Compute**: Heavy (Images + Large Language Model).

*   **Caching (Intermediate Step):**
    *   **Role**: The Bridge. Uses the frozen Stage 1 model to pre-compute and save features for the entire dataset to disk.
    *   **Benefit**: Decouples the heavy visual processing from the planning training.

*   **Stage 2 (Diffusion Planner Training):**
    *   **Role**: The "Brain" (Cerebellum). Trains a lightweight **Diffusion Transformer (DiT)** to generate trajectories based on the cached features.
    *   **Input**: Cached VLM features, Ego Status, History Trajectory.
    *   **Output**: Future trajectory (x, y, heading).
    *   **Compute**: Light & Fast (Vector data only). On TPU v6e, we achieved **~0.26s/step** training speed.

## How to Run

### Stage 1: Vision-Language Pretraining

Use the new `internvl_chat/internvl/train/internvl_chat_finetune_torchax.py` script.

**Example Command:**

```bash
python internvl_chat/internvl/train/internvl_chat_finetune_torchax.py \
  --model_name_or_path "/path/to/ckpt/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --output_dir "work_dirs/ReCogDrive_pretrain_tpu" \
  --meta_path "./shell/data_info/recogdrive_pretrain.json" \
  --force_image_size 448 \
  --max_dynamic_patch 16 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --logging_steps 1 \
  --max_seq_length 12288 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2'
```

### Verified TPU Command (Stage 1)

The following command has been verified to run successfully on a TPU VM (e.g., v6e-4).

```bash
gcloud alpha compute tpus tpu-vm ssh tpu-v6e-4 \
   --project grhuang-02 \
   --zone=us-east5-b \
   --tunnel-through-iap \
   --worker=0 \
   --command=\'docker run \
     --privileged \
     --workdir /recogdrive \
     -v $(pwd)/recogdrive:/recogdrive \
     -e PYTHONPATH=/recogdrive/internvl_chat:$PYTHONPATH \
     --rm --net=host \
     us-east5-docker.pkg.dev/grhuang-02/xiaomi-tpu/recogdrive-tpu:latest \
     python internvl_chat/internvl/train/internvl_chat_finetune_torchax.py \
    --model_name_or_path "models/InternVL3-2B" \
    --conv_style "internvl2_5" \
    --output_dir "tests/work_dirs/test_stage1_tpu" \
    --meta_path "tests/dummy_data/meta.json" \
    --force_image_size 448 \
    --max_dynamic_patch 1 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.0 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 4e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --max_seq_length 2048 \
    --do_train True \
    --grad_checkpoint False \
    --group_by_length False \
    --dynamic_image_size False \
    --use_thumbnail False \
    --ps_version v2\'
```

### Stage 2: Diffusion Planner Training

Use the new `run_training_recogdrive_torchax.py` script.

**Example Command:**

```bash
python navsim/planning/script/run_training_recogdrive_torchax.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.vlm_path='/path/to/pretrain_model' \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    trainer.params.max_epochs=200 \
    experiment_name=training_internvl_agent_dit_tpu \
    train_test_split=navtrain \
    cache_path="/path/to/recogdrive_agent_cache_dir_train_2b" \
    use_cache_without_dataset=True \
    force_cache_computation=False
```

### Verified TPU Command (Stage 2)

The following command has been verified to run successfully on a TPU VM (e.g., v6e-4), utilizing FSDP and JAX JIT compilation. Ensure you have prepared the cache data (dummy or real) in the correct structure before running.

```bash
gcloud alpha compute tpus tpu-vm ssh tpu-v6e-4 \
   --project grhuang-02 \
   --zone=us-east5-b \
   --tunnel-through-iap \
   --worker=0 \
   --command=\'docker run \
     --privileged \
     --workdir /recogdrive \
     -v $(pwd)/recogdrive:/recogdrive \
     -e PYTHONPATH=/recogdrive:/recogdrive/internvl_chat:$PYTHONPATH \
     -e NAVSIM_EXP_ROOT=/recogdrive/tests/work_dirs \
     -e HYDRA_FULL_ERROR=1 \
     --rm --net=host \
     us-east5-docker.pkg.dev/grhuang-02/xiaomi-tpu/recogdrive-tpu:latest \
     /bin/bash -c \"pip install --upgrade hydra-core pytest peft && python navsim/planning/script/run_training_recogdrive_torchax.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.vlm_path=\"models/InternVL3-2B\" \
    agent.cam_type=\"single\" \
    agent.cache_hidden_state=True \
    agent.vlm_type=\"internvl\" \
    agent.dit_type=\"small\" \
    agent.vlm_size=\"small\" \
    agent.sampling_method=\"ddim\" \
    trainer.params.max_epochs=2 \
    experiment_name=training_internvl_agent_dit_tpu \
    train_test_split=navtrain \
    cache_path=\"/recogdrive/tests/dummy_data/recogdrive_agent_cache_dir_train_2b\" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    train_logs=[\"dummy_log\"] \
    val_logs=[\"dummy_log\"] \
    dataloader.params.num_workers=0\"\' 
```

## Notes on Distributed Training

The provided scripts are set up for **single-process** execution (which can utilize multiple TPU cores on a single host via `torchax`'s internal handling or JAX's default behavior). For multi-host TPU pods, you would typically use `jax.distributed.initialize()` and `jax.pmap` or `GSPMD` sharding annotations. `torchax` aims to abstract some of this, but for massive scale, you might need to add explicit sharding annotations to the `train_step`.
