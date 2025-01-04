import subprocess
import os
from ruamel.yaml import YAML
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# importing user defined libraries


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
        "video_process_params": {
            "video_path": "VAE_Analysis/data/Videos/8K_park_stereo.mp4",
            "output_folder": "VAE_Analysis/data/video_dataset/8K_park",
            "is_latent_video": False,
            "top_down_video": True,  # If True, top_down video would be used
            "trim": False,  # If True, trim_start and trim_end would be used
            "trim_start": 0,
            "trim_end": 10,
            "frame_format": ".png",
            "L_R_folder_name": "T_B_frames",
            "num_frames": [0, 150]  # None means all frames
        }
    }

    diff_train_paths = [
        "VAE_Analysis/data/video_dataset/8K_basketball/recon_video",
        "VAE_Analysis/data/video_dataset/8K_sunny/recon_video",
        "VAE_Analysis/data/video_dataset/8K_football/recon_video",
        "VAE_Analysis/data/video_dataset/8K_grass/recon_video",
        "VAE_Analysis/data/video_dataset/8K_park/recon_video",
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
        # print("\nConfig path: ", config_path)
        exp_name = config_path.split("/")[1][:2]
        video_name = f"{exp_name}_orig_120_c_17_libx265_yuv420p.mp4"
        config["video_process_params"]["video_path"] = f"{train_path}/{video_name}"
        config["video_process_params"][
            "output_folder"] = f"{os.path.dirname(train_path)}/reconstructed_frames/libx265_c_17_yuv420p_raw"

        print(config["video_process_params"]["video_path"])
        print(config["video_process_params"]["output_folder"])
        run_script("scripts/video_processing_1.py", config_path, config)


if __name__ == "__main__":
    run_experiments()
