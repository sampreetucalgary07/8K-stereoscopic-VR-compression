# from src.utils.patch_utils import get_patchSize_list
import os
import subprocess
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
            "base_path": "/home/pytorch/personal",
        },
        # "latent_params": {
        #     "frame_path": "VAE_Analysis/data/video_dataset/8K_basketball/L_R_frames",
        #     "model_path": "VAE_Analysis/data/video_dataset/8K_basketball/trained_models/E2_model.pth",
        #     "downsample_factor": 20,
        #     "save_latent_path": "VAE_Analysis/data/video_dataset/8K_basketball/latents/",
        #     "save_latent_folder_name": "latents_tensors",
        #     "is_upsample": False,  # If False, downsample would be used # Downsample or upsample factor
        #     "num_frames": [None, None],  # None means all frames
        #     "tensors_png_path": "VAE_Analysis/data/video_dataset/8K_basketball/latents/",
        #     "tensors_png_folder_name": "latents_tensors_png",
        # },
        # "reconstruction_params": {
        #     "latents_as_img_path": "VAE_Analysis/data/video_dataset/8K_basketball/latents/latents_tensors_png",
        #     "latent_min_max_csv_path": "VAE_Analysis/data/video_dataset/8K_basketball/latents/latents_tensors_png/latent_min_max.csv",
        #     "output_folder": "VAE_Analysis/data/video_dataset/8K_basketball/reconstructed_frames",
        #     "save_folder_name": "raw_frames_from_ReconImg",
        #     "model_path": "VAE_Analysis/data/video_dataset/8K_basketball/trained_models/E2_model.pth",
        #     "upsample_factor": 20,
        #     "num_frames": [None,None],
        # },
        # "evaluation_params": {
        #     "true_frames_path": "VAE_Analysis/data/video_dataset/8K_basketball/L_R_frames",
        #     "recon_frames_path": "VAE_Analysis/data/video_dataset/8K_basketball/reconstructed_frames/raw_frames_from_ReconImg",
        #     "result_json_path": "VAE_Analysis/data/video_dataset/8K_basketball/logs/E2_eval_all.json",
        #     "metrics": ["ssim, psnr"],  # options: ["ssim", "psnr", "lpips"]
        #     "downsample_factor": 20,
        #     "overwrite_prev_json": True,
        #     "patch_wise": False,
        # },
        "parse_csv_params": {
            "overall_avg_metrics": {
                "if_print": True,
                "csv_path": "VAE_Analysis/data/video_dataset/8K_basketball/logs/E2_eval_all.json",
                # options : ["ssim", "psnr", "lpips"]
                "metric_list": ["ssim", "lpips", "psnr"],
                "num_frames": [None, 15],
                "patch_wise": True,
            }
        }
        # "save_video_params": {
        #     "is_latent": False,
        #     "frames_path": "VAE_Analysis/data/video_dataset/8K_park/TL_model_results/E3_model/reconstructed_frames_90/merged_TB_frames",
        #     "output_folder_path": "VAE_Analysis/data/video_dataset/8K_park/TL_model_results/E3_model/recon_video/recon_video_90",
        #     "output_video_name": "recon_E3_90_park",
        #     "video_ext": ".mp4",
        #     "in_fps": 30,
        #     "out_fps": 30,
        #     "video_codec": "libx264",
        #     "pixel_format": "yuv420p",
        #     "img_format": ".png",
        #     "num_frames": [None, 120],  # None means all frames
        #     "crf_value": 21,
        #     "audio_codec": None,
        #     "audio_file_path": None,
        #     "overwrite_prev_video": True,
        # },


    }

    diff_train_paths = [
        "VAE_Analysis/data/video_dataset/8K_basketball",
        "VAE_Analysis/data/video_dataset/8K_sunny",
        "VAE_Analysis/data/video_dataset/8K_football",
        "VAE_Analysis/data/video_dataset/8K_grass",
        "VAE_Analysis/data/video_dataset/8K_park",
    ]

    config_paths = [
        "configs/E2_8K_basketball.yaml",
        "configs/E3_8K_sunny.yaml",
        "configs/E4_8K_football.yaml",
        "configs/E5_8K_grass.yaml",
        "configs/E6_8K_park.yaml",
    ]

    train_flag = True
    for train_path, config_path in zip(diff_train_paths, config_paths):
        print("\nRunning experiment for train path: ", train_path)
        # print("\nConfig path: ", config_path)
        # for percentile in [99,95,90,85,80,75]:
        # for downsample_factor in [40]:# 10,20,30, pending #TO-DO
        #     if train_flag:
        #         str_add = "train"
        #     else:
        #         str_add = "test"
        # print("\nRunning experiment for downsample factor: ",percentile)
        print("\nConfig path: ", config_path)
        # changes in latent_params
        # config["latent_params"]["frame_path"] = train_path
        # config["latent_params"]["model_path"] = os.path.join(
        #     os.path.dirname(train_path),
        #     "trained_models",
        #     f"mE2_tE2_ds_{str(downsample_factor)}_retrain.pth",
        # )
        # config["latent_params"]["downsample_factor"] = downsample_factor
        # config["latent_params"]["save_latent_path"] = os.path.join(
        #     os.path.dirname(train_path), "latents"
        # )
        # config["latent_params"][
        #     "save_latent_folder_name"
        # ] = f"latent_tensor_ds_{str(downsample_factor)}_{str_add}_120_e45"
        # config["latent_params"]["tensors_png_path"] = os.path.join(
        #     os.path.dirname(train_path), "latents"
        # )
        # config["latent_params"][
        #     "tensors_png_folder_name"
        # ] = f"latent_png_ds_{str(downsample_factor)}_{str_add}_120_e45"

        # run_script("scripts/save_latents_6.py", config_path, config)

        # print("\n ----------------- Latent ENDED ----------------- \n")

        # changes in reconstruction_params
        # config["reconstruction_params"]["latents_as_img_path"] = os.path.join(
        #     os.path.dirname(train_path),
        #     "latents",
        #     f"latent_png_ds_{str(downsample_factor)}_{str_add}_120_e45",
        # )
        # config["reconstruction_params"]["latent_min_max_csv_path"] = os.path.join(
        #     os.path.dirname(train_path),
        #     "latents",
        #     f"latent_png_ds_{str(downsample_factor)}_{str_add}_120_e45",
        #     "min_max_values.csv",
        # )
        # config["reconstruction_params"]["output_folder"] = os.path.join(
        #     os.path.dirname(train_path), "reconstructed_frames"
        # )
        # config["reconstruction_params"][
        #     "save_folder_name_raw_tensors"
        # ] = f"raw_latent_frames_ds_{str(downsample_factor)}_{str_add}_120_e45"

        # config["reconstruction_params"][
        #     "save_folder_name_raw_frames"
        # ] = f"raw_frames_ds_{str(downsample_factor)}_{str_add}_120_e45"

        # config["reconstruction_params"]["model_path"] = os.path.join(
        #     os.path.dirname(train_path),
        #     "trained_models",
        #     f"mE2_tE2_ds_{str(downsample_factor)}_retrain.pth",
        # )
        # config["reconstruction_params"]["upsample_factor"] = downsample_factor

        # run_script("scripts/save_frames_7.py", config_path, config)

        # print("\n ----------------- Recon ENDED ----------------- \n")

        # changes in evaluation_params

        # config["evaluation_params"]["true_frames_path"] = os.path.join(
        #     train_path,
        #     "train_frames",
        # )
        # config["evaluation_params"]["recon_frames_path"] = os.path.join(
        #     train_path,
        #     "reconstructed_frames/libx265_c_17_yuv420p_raw/T_B_frames"
        #     # "patch_replaced_40",
        #     # f"frames_{percentile}",
        #     # f"raw_frames_ds_{str(downsample_factor)}_{str_add}_120_e45",
        # )
        # config["evaluation_params"]["result_json_path"] = os.path.join(
        #     train_path,
        #     "logs",
        #     # "patch_replaced_40",
        #     # f"eval_ds_40_{percentile}_{str_add}_lpips.json",
        #     f"eval_ds_libx265_c_17_yuv420p_lpips.json",
        # )
        # config["evaluation_params"]["metrics"] = ["lpips"]
        # config["evaluation_params"]["patch_wise"] = True
        # config["evaluation_params"]["downsample_factor"] = 40
        # config["evaluation_params"]["overwrite_prev_json"] = True

        # print(config["evaluation_params"]["recon_frames_path"])
        # print(config["evaluation_params"]["true_frames_path"])
        # print(config["evaluation_params"]["result_json_path"])
        # print(config["evaluation_params"]["metrics"])
        # print(config["evaluation_params"]["downsample_factor"])
        # print(config["evaluation_params"]["overwrite_prev_json"])

        # run_script("scripts/evaluation_8.py", config_path, config)

        # changes in parse_csv_params
        config["parse_csv_params"]["overall_avg_metrics"]["if_print"] = True
        config["parse_csv_params"]["overall_avg_metrics"]["csv_path"] = os.path.join(
            train_path,
            "logs",
            # "patch_replaced_40",
            f"eval_ds_libx265_c_17_yuv420p_lpips.json",
            # f"eval_ds_40_{str(percentile)}_{str_add}.json",
            # f"eval_ds_{str(downsample_factor)}_{str_add}_120_e45_lpips.json",
        )
        config["parse_csv_params"]["overall_avg_metrics"]["metric_list"] = ["lpips"]
        config["parse_csv_params"]["overall_avg_metrics"]["patch_wise"] = True
        # print(config["parse_csv_params"]["overall_avg_metrics"]["csv_path"])
        config["parse_csv_params"]["overall_avg_metrics"]["num_frames"] = [
            None, None]

        run_script("scripts/parse_csv_10.py", config_path, config)

        # changes in save_video_params
        # config["save_video_params"]["frames_path"] = os.path.join(
        #     train_path,
        #     "frames"
        # )
        # config["save_video_params"]["output_folder_path"] = os.path.join(
        #     train_path,
        #     "recon_video",
        # )
        # config["save_video_params"]["crf_value"] = 30
        # config["save_video_params"]["num_frames"] = [None, 120]
        # config["save_video_params"]["video_codec"] = "libx264"
        # config["save_video_params"]["pixel_format"] = "yuv420p"
        # crf_val = config["save_video_params"]["crf_value"]
        # n_frame = config["save_video_params"]["num_frames"][1]
        # codec = config["save_video_params"]["video_codec"]
        # bit = config["save_video_params"]["pixel_format"]
        # config["save_video_params"][
        #     "output_video_name"] = f"{config_path.split('/')[1][:2]}_orig_{n_frame}_c_{crf_val}_{codec}_{bit}"

        # # print(config)
        # run_script("scripts/save_video_9.py", config_path, config)
        # print(
        #     "\n ----------------- Recon video ENDED ----------------- \n"
        # )


if __name__ == "__main__":
    run_experiments()
