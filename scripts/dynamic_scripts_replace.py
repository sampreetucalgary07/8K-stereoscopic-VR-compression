# Importing required libraries
import os
import subprocess
from ruamel.yaml import YAML
import sys


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
        "patch_replace_params": {
            "input_true_frames_path": "VAE_Analysis/data/video_dataset/8K_sunny/train_frames",
            "input_pred_frames_path": "VAE_Analysis/data/video_dataset/8K_sunny/reconstructed_frames/raw_frames_ds_40_train_120",
            "eval_json_path": "VAE_Analysis/data/video_dataset/8K_sunny/logs/eval_ds_40_train_lpips.json",
            "output_raw_path": "VAE_Analysis/data/video_dataset/8K_sunny/reconstructed_frames/patch_replaced_40/patces",
            "output_frames_path": "VAE_Analysis/data/video_dataset/8K_sunny/reconstructed_frames/patch_replaced_40/frames",
            "output_video_path": "VAE_Analysis/data/video_dataset/8K_sunny/recon_video/patches_replaced_40",
            "output_video_name": "E3_patch_replace_40",
            "overwrite": True,
            "percentile_value": 99,
            "img_dim_H": 3840,
            "img_dim_W": 7680,
            "downsample_factor": 40,
            "metric": "lpips",  # options: "ssim", "psnr", "lpips"
            "num_frames": [None, 10]  # None means all frames
        }

    }

    downsample_factor = 40
    diff_train_paths = [
        f"VAE_Analysis/data/video_dataset/8K_basketball/train_frames",
        f"VAE_Analysis/data/video_dataset/8K_sunny/train_frames",
        f"VAE_Analysis/data/video_dataset/8K_football/train_frames",
        f"VAE_Analysis/data/video_dataset/8K_grass/train_frames",
        f"VAE_Analysis/data/video_dataset/8K_park/train_frames",

    ]
    config_paths = [
        "configs/E2_8K_basketball.yaml",
        "configs/E3_8K_sunny.yaml",
        "configs/E4_8K_football.yaml",
        "configs/E5_8K_grass.yaml",
        "configs/E6_8K_park.yaml",
    ]
    for train_path, config_path in zip(diff_train_paths, config_paths):
        for percentile_value in [99, 95, 90, 85, 80, 75]:
            print("\nRunning experiment for train path: ", train_path)
            print("\nUpdating config file: ", config_path)
            print("\nPercentile value: ", percentile_value)

            config["patch_replace_params"]["input_true_frames_path"] = train_path
            config["patch_replace_params"]["input_pred_frames_path"] = train_path.replace("train_frames",
                                                                                          f"reconstructed_frames/raw_frames_ds_{downsample_factor}_train_120")
            config["patch_replace_params"]["eval_json_path"] = train_path.replace("train_frames",
                                                                                  f"logs/eval_ds_{downsample_factor}_train_120.json")
            config["patch_replace_params"]["output_raw_path"] = train_path.replace("train_frames",
                                                                                   f"reconstructed_frames/patch_replaced_{downsample_factor}/patches_{percentile_value}")
            config["patch_replace_params"]["output_frames_path"] = train_path.replace("train_frames",
                                                                                      f"reconstructed_frames/patch_replaced_{downsample_factor}/frames_{percentile_value}")
            config["patch_replace_params"]["output_video_path"] = train_path.replace("train_frames",
                                                                                     f"recon_video/patches_replaced_{downsample_factor}/video_{percentile_value}")
            config["patch_replace_params"]["output_video_name"] = config_path.split(
                "/")[1][:2] + f"_patch_replace_{downsample_factor}_{percentile_value}"
            config["patch_replace_params"]["percentile_value"] = percentile_value
            config["patch_replace_params"]["downsample_factor"] = downsample_factor
            # print("\nConfig: ", config)

            run_script("scripts/patch_replace_12.py", config_path, config)

            print("\n -------------------------------------------------")


if __name__ == "__main__":
    run_experiments()
