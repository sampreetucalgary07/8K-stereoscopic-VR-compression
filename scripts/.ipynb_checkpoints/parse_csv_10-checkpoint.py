### importing standard libraries
import yaml
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
### importing user defined libraries
from src.eval_csv import (
    parse_patch_csv_all,
    parse_losses_csv,
    parse_plot_frame_wise_loss,
    overall_avg_metrics,
)


def parse_csv(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]
    save_heatmaps = config["parse_csv_params"]["heatmaps"]["if_save"]
    if save_heatmaps:
        csv_path = os.path.join(
            base_path, config["parse_csv_params"]["heatmaps"]["csv_path"]
        )
        output_folder = os.path.join(
            base_path, config["parse_csv_params"]["heatmaps"]["output_folder"]
        )
        num_frames = config["parse_csv_params"]["heatmaps"]["num_frames"]
        save_format = config["parse_csv_params"]["heatmaps"]["save_format"]
        downsample_factor = config["parse_csv_params"]["heatmaps"]["downsample_factor"]
        metric = config["parse_csv_params"]["heatmaps"]["metric"]

        print(
            "\n ----------------- Parsing of csv for heatmaps Started ----------------- \n"
        )
        parse_patch_csv_all(
            csv_path,
            output_folder,
            downsample_factor,
            num_frames,
            save_format,
            metric,
        )

        print("\n ----------------- Parsing Completed ----------------- \n")
        print(f" Heatmaps saved at: {output_folder}")

    save_losses_csv = config["parse_csv_params"]["model_losses"]["if_save"]
    if save_losses_csv:
        csv_path = os.path.join(
            base_path, config["parse_csv_params"]["model_losses"]["csv_path"]
        )
        output_folder = os.path.join(
            base_path, config["parse_csv_params"]["model_losses"]["output_folder"]
        )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        output_name = config["parse_csv_params"]["model_losses"]["output_name"]

        print(
            "\n ----------------- Parsing of csv for losses Started ----------------- \n"
        )
        parse_losses_csv(csv_path, output_folder, output_name)
        print("\n ----------------- Parsing Completed ----------------- \n")
        print(f" Losses csv saved at: {output_folder}")

    frame_wise_loss = config["parse_csv_params"]["plot_frame_wise_loss"]["if_save"]
    if frame_wise_loss:
        csv_path = os.path.join(
            base_path, config["parse_csv_params"]["plot_frame_wise_loss"]["csv_path"]
        )
        output_folder = os.path.join(
            base_path,
            config["parse_csv_params"]["plot_frame_wise_loss"]["output_folder"],
        )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        output_name = config["parse_csv_params"]["plot_frame_wise_loss"]["output_name"]
        downsample_factor = config["parse_csv_params"]["plot_frame_wise_loss"][
            "downsample_factor"
        ]
        num_train_frames = config["parse_csv_params"]["plot_frame_wise_loss"][
            "num_train_frames"
        ]
        metric = config["parse_csv_params"]["plot_frame_wise_loss"]["metric"]
        is_top_down = config["parse_csv_params"]["plot_frame_wise_loss"]["is_top_down"]
        img_format = config["parse_csv_params"]["plot_frame_wise_loss"]["img_format"]

        print(
            "\n ----------------- Parsing of csv for frame wise losses Started ----------------- \n"
        )
        parse_plot_frame_wise_loss(
            csv_path,
            output_folder,
            output_name,
            downsample_factor,
            num_train_frames,
            metric,
            is_top_down,
            img_format,
        )
        print("\n ----------------- Parsing Completed ----------------- \n")
        print(f" Frame wise losses saved at: {output_folder}")
    overall_scene_metrics = config["parse_csv_params"]["overall_avg_metrics"][
        "if_print"
    ]
    num_frames = config["parse_csv_params"]["overall_avg_metrics"]["num_frames"]
    if overall_scene_metrics:
        csv_path = os.path.join(
            base_path, config["parse_csv_params"]["overall_avg_metrics"]["csv_path"]
        )
        metric_list = config["parse_csv_params"]["overall_avg_metrics"]["metric_list"]
        overall_avg_metrics(csv_path, metric_list, num_frames)


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
    parse_csv(args)
