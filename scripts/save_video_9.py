""" This script creats the final video from the reconstructed frames """

### importing standard libraries
import yaml
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

### importing user defined libraries
from src.frames_to_video import frames_to_video_ffmpeg, latents_to_video_ffmpeg


def save_videos(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
    base_path = config["default_params"]["base_path"]
    frames_path = os.path.join(base_path, config["save_video_params"]["frames_path"])
    is_latent = config["save_video_params"]["is_latent"]
    output_folder_path = os.path.join(
        base_path, config["save_video_params"]["output_folder_path"]
    )
    output_video_name = config["save_video_params"]["output_video_name"]
    video_ext = config["save_video_params"]["video_ext"]
    in_fps = config["save_video_params"]["in_fps"]
    out_fps = config["save_video_params"]["out_fps"]
    video_codec = config["save_video_params"]["video_codec"]
    pixel_format = config["save_video_params"]["pixel_format"]
    crf_value = config["save_video_params"]["crf_value"]
    img_format = config["save_video_params"]["img_format"]
    audio_codec = config["save_video_params"]["audio_codec"]
    audio_file_path = config["save_video_params"]["audio_file_path"]
    if audio_file_path is not None:
        audio_file_path = os.path.join(base_path, audio_file_path)
    overwrite_prev_video = config["save_video_params"]["overwrite_prev_video"]
    num_frames = config["save_video_params"]["num_frames"]

    # Generate video from the frames
    if is_latent:
        latents_to_video_ffmpeg(
            frames_path,
            output_video_name,
            video_ext,
            in_fps,
            img_format,
            num_frames,
            output_folder_path,
            overwrite_prev_video,
        )
    else:
        frames_to_video_ffmpeg(
            frames_path,
            output_video_name,
            video_ext,
            in_fps,
            out_fps,
            video_codec,
            pixel_format,
            img_format,
            crf_value,
            num_frames,
            audio_codec,
            audio_file_path,
            output_folder_path,
            overwrite_prev_video,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patches and filter them")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/E1_config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()
    save_videos(args)
