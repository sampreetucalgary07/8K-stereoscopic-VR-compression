""" This script is used to build train and test dataset from the L and R frames """

### importing standard libraries
import yaml
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
### importing user defined libraries
from src.LR_frames_process import LR_train_test_split


def build_dataset(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]

    dataset_config = config["dataset_params"]

    L_R_frames_path = os.path.join(base_path, dataset_config["L_R_frames_path"])
    train_ratio = dataset_config["train_ratio"]
    output_folder = os.path.join(base_path, dataset_config["output_folder"])
    img_format = dataset_config["img_format"]
    seperate_L_R = dataset_config["seperate_L_R"]

    # Split the L and R frames into train and test
    LR_train_test_split(
        LR_frames_path=L_R_frames_path,
        perc_of_train=train_ratio,
        num_images=-0.5,  # -0.5 means all images
        img_format=img_format,
        out_folder_path=output_folder,
        seperate_L_R=seperate_L_R,
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
        help="Path to the config file",
    )
    args = parser.parse_args()
    build_dataset(args)
