# RE-training the model

### importing standard libraries
import yaml
import torch
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import shutil
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
from src.utils.log_utils import append_to_csv
from src.model_func import SaveBestModel, loadModel
from src.utils.file_utils import check_image_size
from src.train_model import train_vae


def retrain_model(args):
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

    model_class = VAE(dataLoader_config["img_channels"], model_config).to(device)

    # retrain parameters
    existing_model_path = os.path.join(base_path, config["retrain_params"]["existing_model_path"])
    copy_model = config["retrain_params"]["copy_model"]
    model_copy_path = os.path.join(base_path, config["retrain_params"]["model_copy_path"])
    # copy the base model to the model_copy_path
    if copy_model:
        os.makedirs(os.path.dirname(model_copy_path), exist_ok=True)
        if os.path.exists(model_copy_path):
            os.remove(model_copy_path)
            print("Previously saved model deleted at the path")
        shutil.copy(existing_model_path, model_copy_path)
        print("\n ----- Model copied successfully ----- \n")
        print("\n ----- Model copied to: ", model_copy_path, " ----- \n")
    else:
        model_copy_path = existing_model_path
        print("\n ----- Model not copied ----- \n")

    if model_copy_path.endswith(".pth") == False:
        raise ValueError("Please provide a valid model path")
    retrain_epochs = config["retrain_params"]["retrain_epochs"]
    save_csv_path = os.path.join(base_path, config["retrain_params"]["log_path"])
    save_model_path = os.path.join(
        base_path, config["retrain_params"]["save_model_path"]
    )
    save_model_name = config["retrain_params"]["save_model_name"]
    save_model_as_pth = config["retrain_params"]["save_model_as_pth"]

    print("\n ----- RETRAINING CONFIGURATION -------\n")
    print("Model path: ", model_copy_path)
    print("Retrain epochs: ", retrain_epochs)
    print("Save csv path: ", save_csv_path)
    print("Save model path: ", save_model_path)
    print("Save model name: ", save_model_name)
    print("Is model save in .pth: ", save_model_as_pth)

    print("\n ----- MODEL CONFIGURATION -------\n   ")
    random_frame = torch.rand(
        dataLoader_config["batch_size"],
        dataLoader_config["img_channels"],
        frame_size[1],  # type: ignore
        frame_size[0],  # type: ignore
    )
    with torch.no_grad():
        _, encode_out = model_class(random_frame.to(device))
    print("Input frames size: ", random_frame.shape)
    print("Output Latents size: ", encode_out.shape)

    #discrminator_class = Discriminator(dataLoader_config["img_channels"]).to(device)
    # Loading the existing model #
    (
        retrain_model,
        optimizer_g,
        optimizer_d,
        discriminator,
        epoch,
        losses_used,
        best_loss_value,
    ) = loadModel( # type: ignore
         model_copy_path,
        model_class,
        Discriminator(im_channels=3),
        pth=True,
        device=device,
    )
    recon_criterion = torch.nn.MSELoss()  # L1/L2 loss for Reconstruction
    disc_criterion = torch.nn.BCEWithLogitsLoss()  # Disc Loss can even be BCEWithLogits
    lpips_model = LPIPS().eval().to(device)

    print("\n ----- Loaded Model INFO ----- \n")
    print(f"Model trained for {epoch} epochs")
    print(f"Model losses: {losses_used}")
    print(f"Model loaded from: {model_copy_path}")
    print(f"Model loaded successfully")
    print((f"Model set to device: {device}"))
    

    retrain_model.train()
    print(f"Model set into training mode \n")

    train_vae(
        retrain_model,
        dataset,
        (epoch, epoch + config["retrain_params"]["retrain_epochs"]),
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
        retrain_best_loss=best_loss_value, # type: ignore
        tl_loss = None,
    )

    print("\n ----- RETRAINING COMPLETED ----- \n")


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
    retrain_model(args)
