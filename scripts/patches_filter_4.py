""" This script creats patches from a dataset and applies filters on it """

### importing standard libraries
import yaml
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

### importing user defined libraries
from src.patches import (
    filter_patches_ssim,
    filter_patches_sample,
    filter_patches_ssim_sample,
)


def patches_filter(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
    base_path = config["default_params"]["base_path"]

    # Extract the config parameters #s
    filter_patch_config = config["filter_patch_params"]
    input_folder = os.path.join(base_path, filter_patch_config["input_folder"])
    output_folder = os.path.join(base_path, filter_patch_config["output_folder"])
    output_folder_name = filter_patch_config["output_folder_name"]
    patch_format = filter_patch_config["patch_format"]
    ssim_threshold = filter_patch_config["ssim_threshold"]
    sample_factor = filter_patch_config["sample_factor"]
    filter_type = filter_patch_config["filter_type"]
    num_patches = filter_patch_config["num_patches"]
    inter_filter = filter_patch_config["inter_filter"]
    if inter_filter == True:
        inter_ssim_threshold = filter_patch_config["inter_ssim_threshold"]
        inter_sample_factor = filter_patch_config["inter_sample_factor"]
    else:
        inter_ssim_threshold = 1.0
        inter_sample_factor = 1

    if filter_type == "ssim":
        # Filter patches based on SSIM
        filter_patches_ssim(
            input_folder_path=input_folder,
            num_patches=num_patches,
            patch_format=patch_format,
            dest_path=output_folder,
            folder_name=output_folder_name,
            threshold=ssim_threshold,
            inter_threshold=inter_ssim_threshold,
        )
    elif filter_type == "sample":
        # Filter patches based on sampling
        filter_patches_sample(
            input_folder,
            num_patches,
            patch_format,
            output_folder,
            output_folder_name,
            sample_factor,
            inter_sample_factor=inter_sample_factor,
        )
    elif filter_type == "both":
        # Filter patches based on SSIM and sampling
        filter_obj = filter_patches_ssim_sample(
            input_folder,
            num_patches,
            patch_format,
            output_folder,
            output_folder_name,
            ssim_threshold,
            sample_factor,
            inter_ssim_threshold,
            inter_sample_factor,
        )
        filter_obj.patch_filter()
        if inter_filter == True:
            filter_obj.inter_patch_filter()
        filter_obj.get_dest_path_info()

    else:
        raise ValueError("Invalid filter type. Please check the config file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patches and filter them")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/E1_config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()
    patches_filter(args)
