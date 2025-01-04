# importing standard libraries
import yaml
import torch
import shutil
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import sys
from GPUtil import showUtilization as gpu_usage
(gpu_usage())
torch.cuda.empty_cache()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# importing user defined libraries
from src.frames_to_video import patches_to_video_ffmpeg
from src.dataLoader import createTorchDataset
from src.utils.file_utils import make_folder
from src.patch_replace import PatchReplace

def patch_replacement(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]
    input_true_frames_path = os.path.join(
        base_path, config["patch_replace_params"]["input_true_frames_path"])
    input_pred_frames_path = os.path.join(
        base_path, config["patch_replace_params"]["input_pred_frames_path"])
    eval_json_path = os.path.join(
        base_path, config["patch_replace_params"]["eval_json_path"])
    output_raw_path = os.path.join(
        base_path, config["patch_replace_params"]["output_raw_path"])
    output_video_path = os.path.join(
        base_path, config["patch_replace_params"]["output_video_path"])
    output_frames_path = os.path.join(
        base_path, config["patch_replace_params"]["output_frames_path"])
    output_video_name = config["patch_replace_params"]["output_video_name"]
    num_frames = config["patch_replace_params"]["num_frames"]
    percentile_value = config["patch_replace_params"]["percentile_value"]
    img_dim = (config["patch_replace_params"]["img_dim_H"],
               config["patch_replace_params"]["img_dim_W"])
    metric = config["patch_replace_params"]["metric"]
    downsample_factor = config["patch_replace_params"]["downsample_factor"]
    overwrite = config["patch_replace_params"]["overwrite"]
    if not os.path.exists(output_raw_path):
        os.makedirs(output_raw_path, exist_ok=True)
        print(f"\nCreated output folder at: {output_raw_path}")
    elif os.path.exists(output_raw_path) and overwrite:
        shutil.rmtree(output_raw_path)
        os.makedirs(output_raw_path, exist_ok=True)
        print(
            f"\nDeleted existing folder and created new folder at: {output_raw_path}")
    elif os.path.exists(output_raw_path) and not overwrite:
        raise ValueError(
            f"\nOutput folder already exists at: {output_raw_path}. Set overwrite=True to delete the folder and create a new one!"
        )
    
    if not os.path.exists(output_frames_path):
        os.makedirs(output_frames_path, exist_ok=True)
        print(f"\nCreated output folder at: {output_frames_path}")

    elif os.path.exists(output_frames_path) and overwrite:
        shutil.rmtree(output_frames_path)
        os.makedirs(output_frames_path, exist_ok=True)
        print(
            f"\nDeleted existing folder and created new folder at: {output_frames_path}")
    elif os.path.exists(output_frames_path) and not overwrite:
        raise ValueError(
            f"\nOutput folder already exists at: {output_frames_path}. Set overwrite=True to delete the folder and create a new one!"
        )
    # Replace the patches in the frames
    PatchReplace(input_true_frames_path,input_pred_frames_path, percentile_value, eval_json_path,
                 output_raw_path,output_frames_path, output_video_path, num_frames,
                 metric,
                 img_dim,
                 downsample_factor)

    # create video from the original patches

    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path, exist_ok=True)
        print(f"\nCreated output folder at: {output_video_path}")

    # patches_to_video_ffmpeg(
    #     frames_path = output_raw_path,
    #     output_video_name = output_video_name,
    #     video_ext = ".mp4",
    #     in_fps = 30,
    #     out_fps = 30,
    #     video_codec = "libx264",
    #     pixel_format = "yuv420p",
    #     img_format=".png",
    #     crf_value=23,
    #     num_frames = [None, None],
    #     audio_codec=None,
    #     audio_path=None,
    #     out_folder_path=output_video_path,
    #     overwrite_previous_video=True,
    # )



if __name__ == "__main__":
    base_dir="/home/pytorch/personal/VAE_Analysis"
    parser=argparse.ArgumentParser(description="Arguments for vae training")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(base_dir, "configs", "E1_config.yaml"),
        help="Path to the config file",
    )
    args=parser.parse_args()
    patch_replacement(args)
