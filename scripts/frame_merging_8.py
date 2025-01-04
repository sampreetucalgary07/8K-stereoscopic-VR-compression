import yaml
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
### importing user defined libraries
from src.merge_frames import merge_LR_frames


def frame_merging(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]
    frames_path = os.path.join(base_path, config["merge_frame_params"]["frames_path"])
    is_same_folder = config["merge_frame_params"]["is_same_folder"]
    is_diff_folder = config["merge_frame_params"]["is_diff_folder"]
    diff_folder_path = os.path.join(
        base_path, config["merge_frame_params"]["diff_folder_path"]
    )
    is_top_down = config["merge_frame_params"]["is_top_down"]
    output_folder_path = os.path.join(
        base_path, config["merge_frame_params"]["output_folder"]
    )
    num_frames = config["merge_frame_params"]["num_frames"]
    output_folder_name = config["merge_frame_params"]["output_folder_name"]
    img_format = config["merge_frame_params"]["img_format"]

    # is_same_folder and is_diff_folder cannot be True at the same time
    if is_same_folder and is_diff_folder:
        raise ValueError(
            "is_same_folder and is_diff_folder cannot be True at the same time"
        )

    if is_same_folder == False and is_diff_folder == False:
        raise ValueError(
            "is_same_folder and is_diff_folder cannot be False at the same time"
        )

    print("\n ----------------- Merging of frames Started ----------------- \n")
    merged_frames_path = merge_LR_frames(
        frames_path,
        output_folder_name,
        is_diff_folder,
        diff_folder_path,
        img_format,
        output_folder_path,
        is_top_down,
        num_frames,
        return_dest_path=True,
    )
    print("\n ----------------- Merging Completed ----------------- \n")
    print(f" Merged frames saved at: {merged_frames_path}")


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
    frame_merging(args)
