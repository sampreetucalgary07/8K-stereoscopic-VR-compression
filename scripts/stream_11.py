import yaml
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
### importing user defined libraries
from src.stream_options import choose_stream_options


def latent_stream(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract the config parameters #

    base_path = config["default_params"]["base_path"]

    latent_tensors_path = os.path.join(
        base_path, config["stream_params"]["latent_tensors_path"]
    )
    save_folder_path = os.path.join(
        base_path, config["stream_params"]["save_folder_path"]
    )
    stream_video_name = config["stream_params"]["stream_video_name"]
    stream_video_ext = config["stream_params"]["stream_video_ext"]
    is_top_down = config["stream_params"]["is_top_down"]
    img_format = config["stream_params"]["img_format"]
    stream_option = config["stream_params"]["stream_option"]
    num_frames = config["stream_params"]["num_frames"]
    stream_sub_option = config["stream_params"]["stream_sub_option"]
    overwrite = config["stream_params"]["overwrite_prev_video"]

    print("\n ----------------- Latent Stream Started ----------------- \n")

    if stream_option == "simulcast":
        choose_stream_options(
            latent_tensors_path,
            stream_video_name,
            is_top_down,
            img_format,
            stream_video_ext,
            save_folder_path,
            num_frames
        ).seperate(stream_sub_option, overwrite)
    elif stream_option == "frame_compatible":  
        choose_stream_options(
            latent_tensors_path,
            stream_video_name,
            is_top_down,
            img_format,
            stream_video_ext,
            save_folder_path,
            num_frames
        ).merge(stream_sub_option, overwrite)
    elif stream_option == "frame_pack":
        choose_stream_options(
            latent_tensors_path,
            stream_video_name,
            is_top_down,
            img_format,
            stream_video_ext,
            save_folder_path,
            num_frames
        ).alternate(stream_sub_option, overwrite)
    else:
        raise ValueError(
            "Invalid stream option.. Please choose from [simulcast, frame_compatible, frame_pack]"
        )


if __name__ == "__main__":
    base_dir = "/home/pytorch/personal/VAE_Analysis"
    parser = argparse.ArgumentParser(description="Process patches and filter them")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(base_dir, "configs", "E1_config.yaml"),
        help="Path to the config file",
    )

    args = parser.parse_args()
    latent_stream(args)
