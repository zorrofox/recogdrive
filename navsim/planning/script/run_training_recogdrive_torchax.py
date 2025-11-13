from typing import Tuple
from pathlib import Path
import logging
import os
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

# torchax imports
import torchax
import jax
import jax.numpy as jnp
import optax
from torchax.interop import JittableModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0
    ).clone().detach()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()

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
    
    # Move agent to JAX device
    logger.info("Moving agent to JAX device...")
    agent = agent.to('jax')

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
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
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    # Optimizer
    # Use optax instead of PyTorch optimizer
    learning_rate = cfg.agent.lr
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(agent.parameters())

    # Training Step
    def train_step(params, batch, opt_state):
        def loss_fn(params, batch):
            features, targets, tokens_list = batch
            # Functional call to agent
            # We need to wrap the forward + loss computation
            # But agent.compute_loss might be stateless.
            # Let's assume agent.forward is the main stateful part.
            
            # We use functional_call to replace params in agent
            # But we need to call agent.forward AND agent.compute_loss
            
            # We can define a helper function that does both
            def forward_and_loss(features, targets, tokens_list):
                prediction = agent.forward(features, targets, tokens_list)
                loss = agent.compute_loss(features, targets, prediction)
                return loss
            
            # Now functional_call this helper? No, functional_call works on modules.
            # We can functional_call the agent.forward, get prediction, then compute loss.
            
            # This assumes compute_loss doesn't use agent's parameters directly (or if it does, they are constant/buffers)
            # If compute_loss uses learnable params, this will be wrong unless we functional_call it too.
            # ReCogDriveAgent.compute_loss seems to just use predictions and targets.
            
            # However, ReCogDriveAgent.forward calls self.action_head(last_hidden_state, action_inputs)
            # self.action_head is a module.
            
            # So:
            prediction = torch.func.functional_call(agent, params, (features, targets, tokens_list))
            loss = agent.compute_loss(features, targets, prediction)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    train_step_jit = jax.jit(train_step)

    logger.info("Starting Training Loop")
    agent.train()
    
    params = dict(agent.named_parameters())
    
    global_step = 0
    max_epochs = cfg.trainer.params.max_epochs
    
    for epoch in range(max_epochs):
        for batch in train_dataloader:
            features, targets, tokens_list = batch
            
            # Move to JAX device
            features = {k: v.to('jax') for k, v in features.items() if isinstance(v, torch.Tensor)}
            targets = {k: v.to('jax') for k, v in targets.items() if isinstance(v, torch.Tensor)}
            
            batch_input = (features, targets, tokens_list)
            
            params, opt_state, loss = train_step_jit(params, batch_input, opt_state)
            
            if global_step % 10 == 0:
                logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss}")
            
            global_step += 1
            
        # Save checkpoint
        save_path = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch}.pt")
        state = {
            'params': params,
            'opt_state': opt_state,
            'epoch': epoch,
            'global_step': global_step
        }
        torchax.save_checkpoint(state, save_path)
        logger.info(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
