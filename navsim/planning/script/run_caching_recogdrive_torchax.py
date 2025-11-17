from typing import Tuple, List, Dict, Any, Union
from pathlib import Path
import logging
import os
import time
import pickle
import gzip
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

# Hack: Mock torch.jit.is_tracing to True to prevent transformers from executing 
# data-dependent control flow (like checking if attention_mask is all 1s), 
# which causes JAX ConcretizationTypeError during JIT tracing.
torch.jit.is_tracing = lambda: True

# Torchax / JAX
import torchax
import jax
import jax.numpy as jnp
import optax
from torchax import tensor as torchax_tensor
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from functools import partial

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.planning.training.dataset import Dataset
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder, TrajectoryTargetBuilder, format_number
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image
from navsim.agents.recogdrive.recogdrive_backbone import IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN, system_message
from navsim.agents.recogdrive.utils.conversation import get_conv_template
from transformers import AutoTokenizer
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)

def get_collate_fn(tokenizer, num_image_token=256, force_image_size=448):
    def collate_fn(batch):
        pixel_values_list = []
        queries = []
        meta_data = []
        
        missing_img_count = 0
        
        for features, targets, token in batch:
            # 1. Decode Image Path
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 2: image_path_tensor = image_path_tensor.squeeze(0)
            
            chars = [chr(c.item()) for c in image_path_tensor if c.item() != 0]
            image_path = "".join(chars)
            
            # 2. Load Image
            # For dummy run, if path is empty/dummy, generate zeros
            if not image_path or not os.path.exists(image_path):
                missing_img_count += 1
                if missing_img_count <= 5: # Log first 5 missing images
                    logger.warning(f"Image path not found: {image_path}, using dummy zeros.")
                pv = torch.zeros((1, 3, force_image_size, force_image_size), dtype=torch.float32)
            else:
                # Force static size for TPU batching
                # We assume we disable dynamic_image_size for caching efficiency on TPU
                pv = load_image(image_path, input_size=force_image_size, max_num=1).unsqueeze(0)
            
            pixel_values_list.append(pv) # Keep (1, 3, H, W)
            
            # 3. Build Question
            history_trajectory = features["history_trajectory"]
            high_command_one_hot = features["high_command_one_hot"]
            
            navigation_commands = ['turn left', 'go straight', 'turn right']
            cmd_idx = (high_command_one_hot == 1).nonzero(as_tuple=True)[0]
            if len(cmd_idx) > 0:
                command_str = navigation_commands[cmd_idx[0].item()]
            else:
                command_str = "unknown"

            history_str = " ".join([f'   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, {format_number(history_trajectory[i, 1].item())}, {format_number(history_trajectory[i, 2].item())})' for i in range(4)])
            
            prompt = f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{history_str}\n3. Active navigation command: [{command_str.upper()}]"
            output_requirements = "\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
            question = f"{prompt}{output_requirements}"
            
            # 4. Prepare Query Template
            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            
            # Replace <image>
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * 1 + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
            
            meta_data.append({'token': token, 'features': features, 'targets': targets})

        # Batch Tokenize
        # Use right padding to match Stage 1 behavior and avoid SDPA path
        tokenizer.padding_side = 'right'
        max_len = 2800 
        
        model_inputs = tokenizer(
            queries, 
            return_tensors='pt', 
            padding='max_length', 
            max_length=max_len, 
            truncation=True
        )
        
        pixel_values = torch.cat(pixel_values_list, dim=0) # (B, 3, 448, 448)
        image_flags = torch.ones((len(batch),), dtype=torch.long)
        
        # Prepare Position IDs (copied from backbone logic)
        attention_mask = model_inputs['attention_mask']
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return {
            'pixel_values': pixel_values,
            'input_ids': model_inputs['input_ids'],
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'image_flags': image_flags
        }, meta_data

    return collate_fn

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    # Support +is_dummy_run arg
    is_dummy_run = cfg.get("is_dummy_run", False)
    
    # Ensure feature builder returns raw data (paths) instead of computing features
    cfg.agent.cache_hidden_state = False
    
    logger.info("Building Agent and Backbone...")
    
    logger.info(f"Loading Backbone from {cfg.agent.vlm_path}...")
    
    # Force eager attention via Config (Layer 1 defense)
    config = InternVLChatConfig.from_pretrained(cfg.agent.vlm_path)
    config.llm_config.attn_implementation = 'eager'
    
    # Load backbone on CPU manually to avoid AutoModel issues
    model = InternVLChatModel.from_pretrained(
        cfg.agent.vlm_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False, # FlashAttn not supported on TPU
        device_map='cpu'
    ).eval()
    
    # HACK: Force disable SDPA in Qwen2 model to avoid JAX ConcretizationTypeError (Layer 2 defense)
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
         if hasattr(model.language_model.model, '_use_sdpa'):
             logger.info("Forcing Qwen2 _use_sdpa=False for TPU compatibility")
             model.language_model.model._use_sdpa = False
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.agent.vlm_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Configure model
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    
    # Enable Torchax after loading weights
    torchax.enable_globally()
    
    # JAX Setup
    num_devices = len(jax.devices())
    logger.info(f"JAX Devices: {jax.devices()}")
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=('data',))
    
    def get_sharding(x):
        if hasattr(x, 'shape') and len(x.shape) > 0 and x.shape[0] % num_devices == 0:
            return NamedSharding(mesh, P('data'))
        else:
            return NamedSharding(mesh, P())

    def to_sharded(pytree):
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, get_sharding(x)), pytree)

    def torch_to_jax(tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        arr = jnp.array(tensor.detach().cpu().numpy())
        return arr

    logger.info("Sharding model parameters to TPU...")
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    all_state_torch = {**params, **buffers}
    jax_state = to_sharded(jax.tree_util.tree_map(torch_to_jax, all_state_torch))
    
    # JIT Step
    @partial(jax.jit, out_shardings=NamedSharding(mesh, P('data')))
    def inference_step(jax_state, inputs):
        env = torchax.default_env()
        torch_state = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), jax_state)
        torch_inputs = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), inputs)
        
        outputs = torch.func.functional_call(
            model, 
            torch_state, 
            args=(), 
            kwargs={**torch_inputs, 'output_hidden_states': True, 'return_dict': True}
        )
        return outputs.hidden_states[-1].jax()

    logger.info("Building Dataset...")
    
    if is_dummy_run:
        # Dummy dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 10
            def __getitem__(self, idx):
                features = {
                    "image_path_tensor": torch.tensor([0], dtype=torch.long), 
                    "history_trajectory": torch.randn(4, 3),
                    "high_command_one_hot": torch.tensor([1.0, 0.0, 0.0]),
                    "status_feature": torch.randn(8)
                }
                targets = {"trajectory": torch.randn(8, 3)}
                return features, targets, f"token_{idx}"
        dataset = DummyDataset()
    else:
        # Real SceneLoader
        scene_filter = instantiate(cfg.train_test_split.scene_filter)
        if cfg.train_logs:
             scene_filter.log_names = cfg.train_logs
        
        # Manually instantiate builders to avoid loading ReCogDriveAgent (which pulls in diffusers/peft conflicts)
        trajectory_sampling = instantiate(cfg.agent.trajectory_sampling)
        
        # Use standard sensor config for RecogDrive (cameras 0-3 + lidar? No, just cameras?)
        # ReCogDriveAgent uses: SensorConfig.build_all_sensors(include=[0, 1, 2, 3])
        sensor_config = SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

        scene_loader = SceneLoader(
            sensor_blobs_path=Path(cfg.sensor_blobs_path),
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )
        
        feature_builders = [ReCogDriveFeatureBuilder(
            cache_hidden_state=False, 
            model_type=cfg.agent.vlm_type,
            checkpoint_path=cfg.agent.vlm_path,
        )]
        target_builders = [TrajectoryTargetBuilder(trajectory_sampling=trajectory_sampling)]
        
        dataset = Dataset(
            scene_loader=scene_loader,
            feature_builders=feature_builders,
            target_builders=target_builders,
            cache_path=None,
            force_cache_computation=True,
        )

    batch_size = cfg.dataloader.params.batch_size
    if batch_size % num_devices != 0:
        logger.warning(f"Batch size {batch_size} not divisible by {num_devices}. Adjusting...")
        batch_size = (batch_size // num_devices) * num_devices
        if batch_size == 0: batch_size = num_devices
        logger.info(f"Adjusted batch size: {batch_size}")

    # FORCE num_workers=0 and pin_memory=False for TPU compatibility
    logger.info("Forcing num_workers=0 and pin_memory=False for TPU compatibility.")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, # CRITICAL FIX
        pin_memory=False, # CRITICAL FIX
        collate_fn=get_collate_fn(tokenizer),
        drop_last=False 
    )

    logger.info(f"Starting Inference on {len(dataset)} samples...")
    output_dir = Path(cfg.cache_path)
    os.makedirs(output_dir, exist_ok=True)
    
    total_time = 0
    count = 0
    
    # Use tqdm for progress tracking
    for batch_idx, (model_inputs, meta_data) in enumerate(tqdm(dataloader, desc="Caching")):
        t0 = time.time()
        
        current_batch_size = model_inputs['input_ids'].shape[0]
        pad_size = 0
        if current_batch_size % num_devices != 0:
            pad_size = num_devices - (current_batch_size % num_devices)
            for k, v in model_inputs.items():
                padding = v[-1:].repeat(pad_size, *([1]*(v.ndim-1)))
                model_inputs[k] = torch.cat([v, padding], dim=0)

        jax_inputs = jax.tree_util.tree_map(
            lambda x: jax.device_put(torch_to_jax(x), get_sharding(torch_to_jax(x))), 
            model_inputs
        )
        
        if batch_idx == 0:
            logger.info("Compiling JAX inference step... (This may take a few minutes)")
        
        hidden_states = inference_step(jax_state, jax_inputs)
        hidden_states.block_until_ready()
        
        if batch_idx == 0:
            logger.info("Compilation finished.")

        hidden_states_cpu = jax.device_get(hidden_states)
        
        if pad_size > 0:
            hidden_states_cpu = hidden_states_cpu[:-pad_size]
            
        for i, meta in enumerate(meta_data):
            token = meta['token']
            sample_hidden_state = torch.tensor(np.array(hidden_states_cpu[i]))
            
            features = meta['features']
            if 'image_path_tensor' in features:
                del features['image_path_tensor']
            features['last_hidden_state'] = sample_hidden_state.float()
            
            targets = meta['targets']
            
            log_name = "dummy_log"
            if not is_dummy_run:
                 scene = dataset._scene_loader.get_scene_from_token(token)
                 log_name = scene.scene_metadata.log_name

            token_dir = output_dir / log_name / token
            os.makedirs(token_dir, exist_ok=True)
            
            dump_feature_target_to_pickle(token_dir / "internvl_feature.gz", features)
            dump_feature_target_to_pickle(token_dir / "trajectory_target.gz", targets)
            
        t1 = time.time()
        step_time = t1 - t0
        total_time += step_time
        count += current_batch_size
        
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}: {step_time:.2f}s ({current_batch_size} samples). Avg: {total_time/count:.4f} s/sample")

    logger.info("Caching Complete.")

if __name__ == "__main__":
    main()
