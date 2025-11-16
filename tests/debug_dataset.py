
import os
from navsim.planning.training.dataset import CacheOnlyDataset
cache_path = "/recogdrive/tests/dummy_data/recogdrive_agent_cache_dir_train_2b"
log_names = ["dummy_log"]
try:
    print(f"Files: {os.listdir(os.path.join(cache_path, 'dummy_log'))}")
    dataset = CacheOnlyDataset(cache_path=cache_path, log_names=log_names, feature_builders=[], target_builders=[])
    print(f"Dataset length: {len(dataset)}")
except Exception as e:
    print(f"Error: {e}")
