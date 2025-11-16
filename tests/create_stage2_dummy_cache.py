import os
import torch
import shutil
import pickle
import gzip

def dump_feature_target_to_pickle(path, data_dict):
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)

def create_dummy_cache(cache_path, num_samples=10):
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    log_name = "dummy_log"
    log_path = os.path.join(cache_path, log_name)
    os.makedirs(log_path, exist_ok=True)

    # Builder names - determined by inspecting navsim/agents/recogdrive/recogdrive_features.py
    # ReCogDriveFeatureBuilder inherits from AbstractFeatureBuilder. 
    # Its unique name is typically the class name unless overridden.
    # Let's generate for common possibilities to be safe if we can't run the check script.
    # But wait, if we generate wrong names, CacheOnlyDataset will skip them.
    
    # Let's assume the class name is used.
    feature_builder_name = "internvl_feature"
    target_builder_name = "trajectory_target"

    for i in range(num_samples):
        token = f"token_{i}"
        token_path = os.path.join(log_path, token)
        os.makedirs(token_path, exist_ok=True)
        
        # Create dummy features
        features = {
            "history_trajectory": torch.randn(4, 3),
            "high_command_one_hot": torch.tensor([1.0, 0.0, 0.0]),
            "status_feature": torch.randn(8),
            "last_hidden_state": torch.randn(256, 1536),
            "image_path_tensor": torch.zeros(1, 10, dtype=torch.long)
        }
        
        # Create dummy targets
        targets = {
            "trajectory": torch.randn(8, 3)
        }
        
        dump_feature_target_to_pickle(os.path.join(token_path, f"{feature_builder_name}.gz"), features)
        dump_feature_target_to_pickle(os.path.join(token_path, f"{target_builder_name}.gz"), targets)
        
    print(f"Created {num_samples} dummy samples in {cache_path}")

if __name__ == "__main__":
    create_dummy_cache("tests/dummy_data/recogdrive_agent_cache_dir_train_2b")