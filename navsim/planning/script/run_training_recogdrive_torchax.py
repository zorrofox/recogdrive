from typing import Tuple
from pathlib import Path
import logging
import os
import time
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict
from functools import partial

# torchax imports
import torchax
import jax
import jax.numpy as jnp
import optax
from torchax.interop import JittableModule
from torchax import tensor as torchax_tensor
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0)
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0)
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0)

    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0
    ).clone().detach()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0)

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'last_hidden_state': last_hidden_state,
        'status_feature': status_feature
    }

    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    # Enable torchax globally
    torchax.enable_globally()
    
    # Seed
    torch.manual_seed(cfg.seed)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    
    # Ensure agent is in training mode
    agent.train()

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        logger.info(f"train_logs type: {type(cfg.train_logs)}, value: {cfg.train_logs}")
        logger.info(f"val_logs type: {type(cfg.val_logs)}, value: {cfg.val_logs}")
        
        logger.info(f"Checking cache path: {cfg.cache_path}")
        if len(cfg.train_logs) > 0:
            log_dir = os.path.join(cfg.cache_path, cfg.train_logs[0])
            if os.path.exists(log_dir):
                logger.info(f"Files in {log_dir}: {os.listdir(log_dir)}")
            else:
                logger.info(f"Log dir {log_dir} does not exist!")

        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=list(cfg.train_logs),
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=list(cfg.val_logs),
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    # Disable pin_memory for TPU compatibility
    dataloader_params = dict(cfg.dataloader.params)
    dataloader_params['pin_memory'] = False
    if dataloader_params.get('num_workers', 0) == 0:
        dataloader_params.pop('prefetch_factor', None)
    
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **dataloader_params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **dataloader_params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    # --- JAX Distributed Setup (FSDP) ---
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

    # Optimizer
    learning_rate = cfg.agent.lr
    optimizer = optax.adamw(learning_rate)
    
    # Get initial params and buffers as JAX arrays
    trainable_params_torch = {k: v.detach() for k, v in agent.named_parameters()}
    buffers_torch = {k: v.detach() for k, v in agent.named_buffers()}
    
    def torch_to_jax(tensor):
        is_bf16 = tensor.dtype == torch.bfloat16
        if is_bf16:
            tensor = tensor.float()
        arr = jnp.array(tensor.detach().cpu().numpy())
        if is_bf16:
            arr = arr.astype(jnp.bfloat16)
        return arr

    jax_trainable_params = to_sharded(jax.tree_util.tree_map(torch_to_jax, trainable_params_torch))
    jax_buffers = to_sharded(jax.tree_util.tree_map(torch_to_jax, buffers_torch))
    
    # Initialize optimizer state
    # Note: optimizer is initialized with trainable parameters only
    opt_state_shape = jax.eval_shape(optimizer.init, jax_trainable_params)
    opt_state_sharding = jax.tree_util.tree_map(get_sharding, opt_state_shape)
    
    @partial(jax.jit, out_shardings=opt_state_sharding)
    def init_opt_state(p):
        return optimizer.init(p)

    logger.info("Initializing optimizer state on TPU...")
    opt_state = init_opt_state(jax_trainable_params)

    # Training Step
    params_sharding = jax.tree_util.tree_map(get_sharding, jax_trainable_params)
    buffers_sharding = jax.tree_util.tree_map(get_sharding, jax_buffers)
    
    @partial(jax.jit, 
             out_shardings=(params_sharding, opt_state_sharding, NamedSharding(mesh, P()))) 
    def train_step_jit(jax_trainable_params, jax_buffers, features, targets, opt_state):
        def loss_fn(trainable_params, buffers, features, targets):
            env = torchax.default_env()
            
            # Merge params and buffers
            all_params = {**trainable_params, **buffers}
            
            torch_params = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), all_params)
            torch_features = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), features)
            torch_targets = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), targets)
            
            # Functional call to agent.forward
            # We pass tokens_list=None as we assume grpo=False for now
            predictions = torch.func.functional_call(agent, torch_params, args=(torch_features, torch_targets, None))
            
            # Compute loss
            loss = agent.compute_loss(torch_features, torch_targets, predictions)
            return loss.jax()

        loss, grads = jax.value_and_grad(loss_fn)(jax_trainable_params, jax_buffers, features, targets)
        updates, opt_state = optimizer.update(grads, opt_state, jax_trainable_params)
        new_jax_params = optax.apply_updates(jax_trainable_params, updates)
        return new_jax_params, opt_state, loss

    logger.info("Starting Training Loop")
    
    global_step = 0
    max_epochs = cfg.trainer.params.max_epochs
    
    for epoch in range(max_epochs):
        for batch in train_dataloader:
            step_start_time = time.time()
            
            features, targets, tokens_list = batch
            
            # Preprocess batch: cast to bf16 (if needed) and shard
            # Assuming model is float32 or bfloat16. Let's check agent dtype?
            # For now, just convert to JAX and shard.
            
            def prepare_batch(data_dict):
                new_dict = {}
                for k, v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        # Cast float32 to bfloat16 if desired? Let's keep float32 for safety unless OOM.
                        # Convert to JAX
                        jax_arr = torch_to_jax(v)
                        # Shard
                        new_dict[k] = jax.device_put(jax_arr, get_sharding(jax_arr))
                return new_dict

            features_jax = prepare_batch(features)
            targets_jax = prepare_batch(targets)
            
            jax_trainable_params, opt_state, loss = train_step_jit(jax_trainable_params, jax_buffers, features_jax, targets_jax, opt_state)
            
            loss.block_until_ready()
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if global_step % 10 == 0:
                logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss}, Time: {step_duration:.4f}s")
            
            global_step += 1
            
        # Save checkpoint
        save_path = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch}.pt")
        
        params_cpu = jax.device_get(jax_trainable_params)
        buffers_cpu = jax.device_get(jax_buffers)
        
        env = torchax.default_env()
        # Merge back for saving
        all_params_cpu = {**params_cpu, **buffers_cpu}
        current_torch_params = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), all_params_cpu)
        
        state = {
            'params': current_torch_params,
            'epoch': epoch,
            'global_step': global_step
        }
        torchax.save_checkpoint(state, save_path, step=global_step)
        logger.info(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()