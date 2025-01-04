""" This script is used to extract L and R frames from the streoscopic video """

### importing standard libraries
import argparse
import yaml
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# print(os.getcwd())
### importing user defined libraries
from src.ffmpeg_video_func import (
    video_info,
    video_to_frames,
    trim_video,
    video_to_latents,
)
from src.LR_frames_process import split_LR_frames


# TO-DO later
# extract audio files
def video_processing(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]

    video_config = config["video_process_params"]

    video_path = os.path.join(base_path, video_config["video_path"])
    is_latent_video = video_config["is_latent_video"]
    output_folder = os.path.join(base_path, video_config["output_folder"])
    trim = video_config["trim"]
    trim_start = video_config["trim_start"]
    trim_end = video_config["trim_end"]
    frame_format = video_config["frame_format"]
    L_R_folder_name = video_config["L_R_folder_name"]
    num_frames = video_config["num_frames"]
    top_down_video = video_config["top_down_video"]

    # Show video info:
    fps, _, _, _ = video_info(video_path, return_values=True)  # type: ignore

    # trim the video:
    if trim == True:
        video_path = trim_video(
            video_path,
            trim_start,
            trim_end,
            os.path.dirname(video_path),
            return_dest_path=True,
        )

    # Extract frames from the video
    if is_latent_video:
        saved_frames_path = video_to_latents(
            mp4_file_path=video_path,
            output_folder_name="latents",
            frame_rate=fps,
            out_folder_path=output_folder,
            return_dest_path=True,
        )
    else:
        saved_frames_path = video_to_frames(
            mp4_file_path=video_path,
            output_folder_name="frames",
            frame_rate=fps,
            out_folder_path=output_folder,
            return_dest_path=True,
        )

    # Extract L and R frames from the frames
    split_LR_frames(
        saved_frames_path,
        L_R_folder_name,
        frame_format,
        output_folder,
        top_down_video,
        num_frames,
    )


if __name__ == "__main__":
    base_dir = "/home/pytorch/personal/VAE_Analysis"
    parser = argparse.ArgumentParser(
        description="Process video and extract L and R frames"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(base_dir, "configs", "E1_config.yaml"),
        help="path to the config file",
    )
    args = parser.parse_args()
    video_processing(args)
