""" This script evaluates the model using the given frames """

### importing standard libraries
import yaml
import torch

torch.cuda.empty_cache()
from GPUtil import showUtilization as gpu_usage

(gpu_usage())
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

### importing user defined libraries
from src.eval import eval_frame_all_dataloader


def evaluation(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
    base_path = config["default_params"]["base_path"]
    true_frames_path = os.path.join(
        base_path, config["evaluation_params"]["true_frames_path"]
    )
    recon_frames_path = os.path.join(
        base_path, config["evaluation_params"]["recon_frames_path"]
    )
    result_csv_path = os.path.join(
        base_path, config["evaluation_params"]["result_json_path"]
    )
    # check csv file exists if not create one
    overwrite = config["evaluation_params"]["overwrite_prev_json"]

    if os.path.exists(result_csv_path):
        if overwrite:
            os.remove(result_csv_path)
            print("\n\n------- !! Previous CSV file removed !!------- \n")
        else:
            raise FileExistsError(
                "The result csv file already exists, please change the name or set overwrite_prev_csv to True"
            )
    else:
        os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
        # with open(result_csv_path, "w") as f:
        #     f.write("Contains Results of the evaluation\n")

    metrics_list = config["evaluation_params"]["metrics"]
    img_dim = (
        config["evaluation_params"]["img_dim_H"],
        config["evaluation_params"]["img_dim_W"],
    )
    downsample_factor = config["evaluation_params"]["downsample_factor"]
    num_frames = config["evaluation_params"]["num_frames"]
    patch_wise = config["evaluation_params"]["patch_wise"]
    print("\n ------------ Evaluation Parameters ------------ \n")
    print(f"True Frames Path: {true_frames_path}")
    print(f"Reconstructed Frames Path: {recon_frames_path}")
    print(f"Result CSV Path: {result_csv_path}")
    print(f"Metrics List: {metrics_list}")
    print(f"Image Dimension: {img_dim}")
    print(f"Downsample Factor: {downsample_factor}")
    print(f"Number of Frames: {num_frames}")
    print("\n ------------ Starting Evaluation ------------ \n")
    # Getting the results:

    eval_frame_all_dataloader(
        true_frames_path,
        recon_frames_path,
        result_csv_path,
        img_dim,
        downsample_factor,
        metrics_list,
        num_frames,
        "cuda",
        patch_wise,
    )

    print("\n ------------ Evaluation completed ------------ \n")
    print(f"Results saved at: {result_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patches and filter them")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/E1_config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()
    evaluation(args)
