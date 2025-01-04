""" This script contains methods to evaluate diff csv files"""

# importing standard libraries


from tqdm import tqdm
import csv
import os
import ast
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# importing user defined libraries
from src.utils.eval_utils import read_csv, pd_read_csv, read_json


def parse_patch_line(line_list, downsample_factor, metric="ssim"):
    metric_values = []
    # patch_values = []
    annotations = []
    titles = [line_list["true_frame_path"], line_list["pred_frame_path"]]

    for i in line_list["patch_metric"]:
        for key, value in i.items():
            metric_values.append(value[metric])
            annotations.append(
                str(key.replace("Patch_", "P ")) + "\n" + str(round(value[metric], 3))
            )

    metric_matrix = np.array(metric_values).reshape(
        downsample_factor, downsample_factor
    )
    # patch_matrix = np.array(Patch_values).reshape(downsample_factor, downsample_factor)
    annotations_matrix = np.array(annotations).reshape(
        downsample_factor, downsample_factor
    )
    return metric_matrix, annotations_matrix, titles


def plot_matrix(
    metric_matrix, annotations_matrix, titles, save_path, metric: str, cbar=True
):

    plt.figure(figsize=(26, 13))

    sns.heatmap(
        metric_matrix,
        annot=annotations_matrix,
        fmt="",
        annot_kws={"size": 5},
        cbar=cbar,
    )
    plt.title(f"{metric.upper()} Heatmap for {titles[0]} V/S {titles[1]}", fontsize=14)
    # remove ticks
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_patch_csv_all(
    csv_file_path: str,
    output_folder: str,
    downsample_factor: int,
    num_frames: list,
    save_format: str,
    metric: str,
    img_format=".png",
):
    data_list = read_json(csv_file_path)
    count = 0
    for line_list in tqdm(
        data_list[num_frames[0] :], total=len(data_list), desc="Parsing CSV.. "
    ):
        # if line_list[0].endswith(img_format):

        metric_matrix, annotations_matrix, titles = parse_patch_line(
            line_list, downsample_factor, metric
        )

        save_name = str(line_list["true_frame_path"].split("_")[1].split(".")[0])
        save_path = os.path.join(output_folder, f"heat_{save_name}{save_format}")
        if os.path.exists(save_path):
            os.remove(save_path)

        plot_matrix(
            metric_matrix, annotations_matrix, titles, save_path, metric, cbar=True
        )
        count += 1
        if count == num_frames[1]:
            break


def parse_losses_csv(csv_file_path: str, output_folder: str, output_name: str):

    df = pd_read_csv(csv_file_path)

    plt.plot(df["epoch"], df["recon_loss"], label="Recon Loss")
    plt.plot(df["epoch"], df["perceptual_loss"], label="Perceptual Loss")
    plt.plot(df["epoch"], df["disc_loss"], label="Disc Loss")
    plt.plot(df["epoch"], df["gen_loss"], label="Gen Loss")
    plt.xlabel("Epochs ->")
    plt.ylabel("Losses ->")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.title("Diff. Losses")
    plt.legend()
    save_path = os.path.join(output_folder, output_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_plot_frame_wise_loss(
    csv_file_path: str,
    output_folder: str,
    output_name: str,
    downsample_factor: int,
    num_train_frames: int,
    metric: str,
    is_top_down: bool,
    img_format=".png",
):

    data_list = read_json(csv_file_path)
    if is_top_down:
        ext_1 = "T"
        ext_2 = "B"
    else:
        ext_1 = "L"
        ext_2 = "R"

    metric_mean_values_ext_1 = []
    metric_mean_values_ext_2 = []
    for idx, line_list in enumerate(
        tqdm(data_list, total=len(data_list), desc="Parsing CSV : ")
    ):

        if line_list[0].endswith(ext_1 + img_format):  # LATER : change to ext_1
            metric_matrix, _, _ = parse_patch_line(line_list, downsample_factor, metric)
            metric_mean_values_ext_1.append(np.mean(metric_matrix.flatten()).round(4))
        if line_list[0].endswith(ext_2 + img_format):  # LATER : change to ext_2
            metric_matrix, _, _ = parse_patch_line(line_list, downsample_factor, metric)
            metric_mean_values_ext_2.append(np.mean(metric_matrix.flatten()).round(4))
    # plotting
    plt.figure(figsize=(10, 5))
    plt.plot(metric_mean_values_ext_1, label=f"{ext_1} Frames")
    plt.axvline(x=num_train_frames, color="r", linestyle="--")
    plt.plot(metric_mean_values_ext_2, label=f"{ext_2} Frames")
    plt.title(f"{metric.upper()} of {ext_1} and {ext_2} Frames")
    plt.xlabel("Frame Index")
    plt.ylabel(f"{metric.upper()}")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.legend()
    save_path = os.path.join(output_folder, output_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def overall_avg_metrics(csv_file_path: str, metrics: list, num_frames:list, patch_wise: bool = True):
    data_list = read_json(csv_file_path)
    # print(data_list
    # print(type(data_list))
    # print(len(data_list))
    # print(num_frames)
    for metric in metrics:
        metric_values = []
        for line_list in data_list[num_frames[0]:num_frames[1]]:
            for i in line_list["patch_metric"]:
                if patch_wise:
                    for key, value in i.items():
                        metric_values.append(value[metric])
                else:
                    try:
                        metric_values.append(i[metric])
                    except:
                        pass
        print(f"\nOverall {metric} : {np.mean(metric_values)}")
        print(f"Min {metric} : {np.min(metric_values)}")
        print(f"Max {metric} : {np.max(metric_values)}")
