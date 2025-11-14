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

## How to Run

### Stage 1: Vision-Language Pretraining

Use the new `internvl_chat_finetune_torchax.py` script. You can adapt your existing shell script (`internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh`) to call this python script instead of `torchrun ... internvl_chat_finetune.py`.

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

*Note: DeepSpeed arguments are removed as `torchax`/JAX handles optimization.*

### Verified TPU Command (Stage 1)

The following command has been verified to run successfully on a TPU VM (e.g., v6e-4) using a Docker container:

```bash
gcloud alpha compute tpus tpu-vm ssh tpu-v6e-4 \
   --project grhuang-02 \
   --zone=us-east5-b \
   --tunnel-through-iap \
   --worker=0 \
   --command='docker run \
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
    --max_dynamic_patch 4 \
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
    --dynamic_image_size True \
    --use_thumbnail True \
    --ps_version v2'
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

## Notes on Distributed Training

The provided scripts are set up for **single-process** execution (which can utilize multiple TPU cores on a single host via `torchax`'s internal handling or JAX's default behavior). For multi-host TPU pods, you would typically use `jax.distributed.initialize()` and `jax.pmap` or `GSPMD` sharding annotations. `torchax` aims to abstract some of this, but for massive scale, you might need to add explicit sharding annotations to the `train_step`.
