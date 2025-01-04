#Importing required libraries
import os
import subprocess
from ruamel.yaml import YAML
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# importing user defined libraries
from src.utils.patch_utils import get_patchSize_list

def update_yaml(file_path, new_values):
    """Update the YAML file with new values, preserving comments."""
    yaml = YAML()

    # Load the existing YAML file
    with open(file_path, "r") as file:
        data = yaml.load(file)

    # Update the data with new values
    def update_dict(original, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                original[key] = update_dict(original.get(key, {}), value)
            else:
                original[key] = value
        return original

    updated_data = update_dict(data, new_values)

    yaml.default_flow_style = False
    # Write the updated YAML file, preserving comments
    with open(file_path, "w") as file:
        yaml.dump(updated_data, file)


def run_script(script_name, config_path, config):
    update_yaml(config_path, config)
    subprocess.run(
        [
            "python3",
            script_name,
            "--config_path",
            config_path,
        ]
    )


def run_experiments():

    config = {
        "default_params": {
            "base_path": "/home/pytorch/personal"
        },
        "dataLoader_params": {
            "dataset_path": None,
            "start_frame_num": 0,
            "end_frame_num": None,  # -1 means all frames
            "batch_size": 1,
            "shuffle": True,
            "channel_last": False,
            "img_channels": 3,
        },
        "model_params": {
            "z_channels": 3,
            "down_channels": [128, 256, 512, 512],  # 2x4 x reduction
            "down_sample": [True, True, True],
            "attn_down": [False, False, False],
            "norm_channels": 32,
            "num_down_layers": 2,
            "num_up_layers": 2,
        },
        "training_params": {
            "device": "cuda",
            "disc_start": 15000,
            "num_epochs": 15,
            "save_model_start": 1,
            "save_model_path": "VAE_Analysis/data/video_dataset/8K_basketball/trained_models",
            "save_model_name": "E2_model_dis",  # Don't add .pt or .pth
            "save_model_as_pth": True,
            "save_csv_path": "VAE_Analysis/data/video_dataset/8K_basketball/logs/E2_raw_dis.csv",
        },
    }
    diff_train_paths = [
        "VAE_Analysis/data/video_dataset/8K_basketball/filtered_f_ptchs/inter_patch_filter",
        # "VAE_Analysis/data/video_dataset/8K_sunny/filtered_f_ptchs/inter_patch_filter",
        # "VAE_Analysis/data/video_dataset/8K_football/filtered_f_ptchs/inter_patch_filter",
        # "VAE_Analysis/data/video_dataset/8K_grass/filtered_f_ptchs/inter_patch_filter",
        # "VAE_Analysis/data/video_dataset/8K_park/filtered_f_ptchs/inter_patch_filter"
    ]
    config_paths = [
        "configs/E2_8K_basketball.yaml",
        # "configs/E3_8K_sunny.yaml",
        # "configs/E4_8K_football.yaml",
        # "configs/E5_8K_grass.yaml",
        # "configs/E6_8K_park.yaml",
    ]
    for train_path, config_path in zip(diff_train_paths, config_paths):
        print("\nRunning experiment for train path: ", train_path)
        # print("\nConfig path: ", config_path)
        for downsample_factor in [10]:
            print("\nRunning experiment for downsample factor: ", downsample_factor)
            print("\nConfig path: ", config_path)

            # changes in dataLoader_params
            config["dataLoader_params"]["dataset_path"] = os.path.join(
                config["default_params"]["base_path"], train_path)

            # changes in training_params
            config["training_params"]["disc_start"] = (
                len(
                    os.listdir(
                        os.path.join(
                            config["default_params"]["base_path"],
                            config["dataLoader_params"]["dataset_path"],
                        )
                    )
                )
                * 3
            )
            
            config["training_params"]["save_model_path"] = os.path.join(
                os.path.dirname(os.path.dirname(train_path)), "trained_models"
            )
            config["training_params"][
                "save_model_name"
            ] = f"mE2_tE2_ds_{downsample_factor}"
            config["training_params"]["save_csv_path"] = os.path.join(
                os.path.dirname(os.path.dirname(train_path)),
                f"logs/mE2_tE2_ds_{downsample_factor}.csv",
            )

            run_script("scripts/train_vae_model_5.py", config_path, config)


if __name__ == "__main__":
    run_experiments()
