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
        "stream_params":{
            "latent_tensors_path": "VAE_Analysis/data/video_dataset/8K_basketball/latents/L_R_frames_latent_tensors",
            "is_top_down": True,
            "save_folder_path": "VAE_Analysis/data/video_dataset/8K_basketball/recon_video/latent_stream",
            "stream_video_name": "E2_stream",
            "stream_video_ext": ".mp4",
            "img_format": ".png",
            "stream_option": "alternate",  # 'separate' or 'merge' or 'alternate'
            "num_frames": [None, None],  # None means all frames
            "stream_sub_option": "residual",  # 'residual' or 'raw'
            "overwrite_prev_video": True,
        }
    }

    downsample_factor = 40
    diff_train_paths = [
        f"VAE_Analysis/data/video_dataset/8K_basketball/latents/latent_tensor_ds_{downsample_factor}_train_120",
        f"VAE_Analysis/data/video_dataset/8K_sunny/latents/latent_tensor_ds_{downsample_factor}_train_120",
        f"VAE_Analysis/data/video_dataset/8K_football/latents/latent_tensor_ds_{downsample_factor}_train_120",
        f"VAE_Analysis/data/video_dataset/8K_grass/latents/latent_tensor_ds_{downsample_factor}_train_120",
        f"VAE_Analysis/data/video_dataset/8K_park/latents/latent_tensor_ds_{downsample_factor}_train_120",

    ]
    config_paths = [
        "configs/E2_8K_basketball.yaml",
        "configs/E3_8K_sunny.yaml",
        "configs/E4_8K_football.yaml",
        "configs/E5_8K_grass.yaml",
        "configs/E6_8K_park.yaml",
    ]
    for train_path, config_path in zip(diff_train_paths, config_paths):
        print("\nRunning experiment for train path: ", train_path)
        print("\nUpdating config file: ", config_path)
        for stream_option in ['simulcast', 'frame_compatible', 'frame_pack']:
            for stream_sub_option in ['residual', 'raw']:
                print("\nRunning experiment for stream option: ", stream_option)
                print("\nRunning experiment for stream sub option: ", stream_sub_option)

                config["stream_params"]["latent_tensors_path"] = train_path
                config["stream_params"]["save_folder_path"] = os.path.join(os.path.dirname(os.path.dirname(train_path)), f"recon_video/latent_stream_{downsample_factor}_e45")
                config["stream_params"]["stream_video_name"] = config_path.split("/")[1][:2] + f"_stream_{downsample_factor}_e45_"

                config["stream_params"]["stream_option"] = stream_option
                config["stream_params"]["stream_sub_option"] = stream_sub_option

                run_script("scripts/stream_11.py", config_path, config)
        


if __name__ == "__main__":
    run_experiments()
