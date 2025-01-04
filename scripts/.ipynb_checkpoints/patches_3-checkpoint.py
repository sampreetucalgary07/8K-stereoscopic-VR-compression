""" This script creats patches from a dataset and applies filters on it """

### importing standard libraries
import yaml
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
### importing user defined libraries
from src.patches import frame_to_patches


def patches(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]

    patch_config = config["patch_params"]
    input_dataset = os.path.join(base_path, patch_config["input_dataset"])
    img_dim = (patch_config["img_dim_H"], patch_config["img_dim_W"])
    downsample_factor = patch_config["downsample_factor"]
    output_folder = os.path.join(base_path, patch_config["output_folder"])
    num_images = patch_config["num_frames"]
    device = patch_config["device"]

    # Save patches from the dataset
    frame_to_patches(
        input_dataset,
        img_dim,
        downsample_factor,
        destination_path=output_folder,
        num_images=num_images,
        return_dest_path=False,
        device = device
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
    patches(args)
