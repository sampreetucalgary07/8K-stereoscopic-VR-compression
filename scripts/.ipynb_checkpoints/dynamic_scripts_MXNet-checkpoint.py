import os
import subprocess
from ruamel.yaml import YAML
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
### importing user defined libraries
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
            "base_path": "/home/pytorch/personal",
        },
        "latent_params": {
            "frame_path": "VAE_Analysis/data/video_dataset/8K_basketball/L_R_frames",
            "model_path": "VAE_Analysis/data/video_dataset/8K_basketball/trained_models/E2_model.pth",
            "downsample_factor": 20,
            "save_latent_path": "VAE_Analysis/data/video_dataset/8K_basketball/latents/",
            "save_latent_folder_name": "latents_tensors",
            "is_upsample": False,  # If False, downsample would be used # Downsample or upsample factor
            "num_frames": [None, 15],  # None means all frames
            "tensors_png_path": "VAE_Analysis/data/video_dataset/8K_basketball/latents/",
            "tensors_png_folder_name": "latents_tensors_png",
        },
        "reconstruction_params": {
            "latents_as_img_path": "VAE_Analysis/data/video_dataset/8K_basketball/latents/latents_tensors_png",
            "latent_min_max_csv_path": "VAE_Analysis/data/video_dataset/8K_basketball/latents/latents_tensors_png/latent_min_max.csv",
            "output_folder": "VAE_Analysis/data/video_dataset/8K_basketball/reconstructed_frames",
            "save_folder_name": "raw_frames_from_ReconImg",
            "model_path": "VAE_Analysis/data/video_dataset/8K_basketball/trained_models/E2_model.pth",
            "upsample_factor": 20,
            "num_frames": None,  # -1 means all frames
        },
        "evaluation_params": {
            "true_frames_path": "VAE_Analysis/data/video_dataset/8K_basketball/L_R_frames",
            "recon_frames_path": "VAE_Analysis/data/video_dataset/8K_basketball/reconstructed_frames/raw_frames_from_ReconImg",
            "result_json_path": "VAE_Analysis/data/video_dataset/8K_basketball/logs/E2_eval_all.json",
            "metrics": ["ssim", "psnr"],  # options: ["ssim", "psnr", "lpips"]
            "downsample_factor": 20,
            "overwrite_prev_json": True,
            "patch_wise": False,
        },
    }

    diff_train_paths = [
        "VAE_Analysis/data/video_dataset/8K_basketball/test_frames",
        "VAE_Analysis/data/video_dataset/8K_sunny/test_frames",
        "VAE_Analysis/data/video_dataset/8K_football/test_frames",
        "VAE_Analysis/data/video_dataset/8K_grass/test_frames",
        "VAE_Analysis/data/video_dataset/8K_park/test_frames",
    ]
    config_paths = [
        "configs/E2_8K_basketball.yaml",
        "configs/E3_8K_sunny.yaml",
        "configs/E4_8K_football.yaml",
        "configs/E5_8K_grass.yaml",
        "configs/E6_8K_park.yaml",
    ]

    train_flag = False
    for train_path, config_path in zip(diff_train_paths, config_paths):
        print("\nRunning experiment for train path: ", train_path)
        # print("\nConfig path: ", config_path)
        for downsample_factor in [10, 30, 40]:
            if train_flag:
                str_add = "train"
            else:
                str_add = "test"
            print("\nRunning experiment for downsample factor: ", downsample_factor)
            print("\nConfig path: ", config_path)
            # changes in latent_params
            config["latent_params"]["frame_path"] = train_path
            config["latent_params"]["model_path"] = os.path.join(
                os.path.dirname(train_path),
                "trained_models",
                f"mE2_tE2_ds_{str(downsample_factor)}.pth",
            )
            config["latent_params"]["downsample_factor"] = downsample_factor
            config["latent_params"]["save_latent_path"] = os.path.join(
                os.path.dirname(train_path), "latents"
            )
            config["latent_params"][
                "save_latent_folder_name"
            ] = f"latent_tensor_ds_{str(downsample_factor)}_{str_add}"
            config["latent_params"]["tensors_png_path"] = os.path.join(
                os.path.dirname(train_path), "latents"
            )
            config["latent_params"][
                "tensors_png_folder_name"
            ] = f"latent_png_ds_{str(downsample_factor)}_{str_add}"

            run_script("scripts/save_latents_6.py", config_path, config)

            # changes in reconstruction_params
            config["reconstruction_params"]["latents_as_img_path"] = os.path.join(
                os.path.dirname(train_path),
                "latents",
                f"latent_png_ds_{str(downsample_factor)}_{str_add}",
            )
            config["reconstruction_params"]["latent_min_max_csv_path"] = os.path.join(
                os.path.dirname(train_path),
                "latents",
                f"latent_png_ds_{str(downsample_factor)}_{str_add}",
                "min_max_values.csv",
            )
            config["reconstruction_params"]["output_folder"] = os.path.join(
                os.path.dirname(train_path), "reconstructed_frames"
            )
            config["reconstruction_params"][
                "save_folder_name_raw_tensors"
            ] = f"raw_latent_frames_ds_{str(downsample_factor)}_{str_add}"

            config["reconstruction_params"][
                "save_folder_name_raw_frames"
            ] = f"raw_frames_ds_{str(downsample_factor)}_{str_add}"

            config["reconstruction_params"]["model_path"] = os.path.join(
                os.path.dirname(train_path),
                "trained_models",
                f"mE2_tE2_ds_{str(downsample_factor)}.pth",
            )
            config["reconstruction_params"]["upsample_factor"] = downsample_factor

            run_script("scripts/save_frames_7.py", config_path, config)

            # changes in evaluation_params
            config["evaluation_params"]["true_frames_path"] = train_path
            config["evaluation_params"]["recon_frames_path"] = os.path.join(
                os.path.dirname(train_path),
                "reconstructed_frames",
                f"raw_frames_ds_{str(downsample_factor)}_{str_add}",
            )
            config["evaluation_params"]["result_json_path"] = os.path.join(
                os.path.dirname(train_path),
                "logs",
                f"eval_ds_{str(downsample_factor)}_{str_add}.json",
            )
            config["evaluation_params"]["downsample_factor"] = downsample_factor
            config["evaluation_params"]["overwrite_prev_json"] = True
            config["evaluation_params"]["patch_wise"] = False

            run_script("scripts/evaluation_8.py", config_path, config)


if __name__ == "__main__":
    run_experiments()
