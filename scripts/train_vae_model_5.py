# training the model

### importing standard libraries
import yaml
import torch
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import sys
from GPUtil import showUtilization as gpu_usage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
torch.cuda.empty_cache()
(gpu_usage())
# from torchvision.utils import make_gri

### importing user defined libraries
from src.dataLoader import createTorchDataset
from model.vae_model import VAE
from model.lpips import LPIPS
from model.discriminator import Discriminator
from src.utils.log_utils import init_csv, append_to_csv, csv_dir_check
from src.model_func import SaveBestModel, loadModel
from src.utils.file_utils import check_image_size
from src.train_model import train_vae


def train_model(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]
    dataLoader_config = config["dataLoader_params"]
    model_config = config["model_params"]
    training_config = config["training_params"]
    device = training_config["device"]

    print("\n ----- DATALOADER  CONFIGURATION -------\n")
    print("Dataset path: ", dataLoader_config["dataset_path"])
    print("Start frame number: ", dataLoader_config["start_frame_num"])
    print("End frame number: ", dataLoader_config["end_frame_num"])
    print("Channel last: ", dataLoader_config["channel_last"])
    print("Batch size: ", dataLoader_config["batch_size"])
    print("Shuffle: ", dataLoader_config["shuffle"])
    print("Frame channel: ", dataLoader_config["img_channels"])

    print("\n ----- Checking Directory and Frame Size -------\n")
    frame_size = check_image_size(
        os.path.join(base_path, dataLoader_config["dataset_path"]), return_img_size=True
    )
    # Create the dataset to be trained on #
    sample_dataset = createTorchDataset(
        os.path.join(base_path, dataLoader_config["dataset_path"]),
        start_num_image=dataLoader_config["start_frame_num"],
        end_num_image=dataLoader_config["end_frame_num"],
        channel_last=dataLoader_config["channel_last"],
    )
    dataset = DataLoader(
        sample_dataset,
        batch_size=dataLoader_config["batch_size"],
        shuffle=dataLoader_config["shuffle"],
    )

    # Create the model
    model = VAE(dataLoader_config["img_channels"], model_config).to(device)

    # Model parameters #
    print("\n ----- MODEL CONFIGURATION -------\n   ")
    random_frame = torch.rand(
        dataLoader_config["batch_size"],
        dataLoader_config["img_channels"],
        frame_size[1],  # type: ignore
        frame_size[0],  # type: ignore
    )
    with torch.no_grad():
        _, encode_out = model(random_frame.to(device))
    print("Input frames size: ", random_frame.shape)
    print("Output Latents size: ", encode_out.shape)

    recon_criterion = torch.nn.MSELoss()  # L1/L2 loss for Reconstruction
    disc_criterion = torch.nn.BCEWithLogitsLoss()  # Disc Loss can even be BCEWithLogits
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=3).to(device)
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=training_config["disc_lr"],
        betas=(0.5, 0.999),
    )

    optimizer_g = torch.optim.Adam(
        model.parameters(), lr=training_config["gen_lr"], betas=(0.5, 0.999)
    )

    ## save paths paramters
    save_model_path = os.path.join(base_path, training_config["save_model_path"])
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    save_model_as_pth = training_config["save_model_as_pth"]
    save_model_name = training_config["save_model_name"]
    save_csv_path = os.path.join(base_path, training_config["save_csv_path"])
    csv_dir_check(save_csv_path, True)
    fieldnames = [
        "epoch",
        "recon_loss",
        "perceptual_loss",
        "disc_loss",
        "gen_loss",
        "time_taken",
    ]

    init_csv(save_csv_path, fieldnames)

    model.train()

    train_vae(
        model,
        dataset,
        (0, training_config["num_epochs"]),
        optimizer_g,
        optimizer_d,
        discriminator,
        recon_criterion,
        lpips_model,
        disc_criterion,
        training_config,
        save_model_path,
        save_model_name,
        save_model_as_pth,
        save_csv_path,
    )


if __name__ == "__main__":
    base_dir = "/home/pytorch/personal/VAE_Analysis"
    parser = argparse.ArgumentParser(description="Arguments for vae training")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(base_dir, "configs", "E1_config.yaml"),
        help="Path to the config file",
    )
    args = parser.parse_args()
    train_model(args)
