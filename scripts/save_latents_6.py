"""This script is used to save latents from the given frames using the model"""

### importing standard libraries
import os
import yaml
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

### importing user defined libraries
from src.patches import get_patchSize_list
from src.model_to_latents import save_raw_latent_all, save_tensor_as_png_all
from src.model_func import loadModel
from model.vae_model import VAE
from model.discriminator import Discriminator


def save_latents(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]
    latent_config = config["latent_params"]
    dataLoader_config = config["dataLoader_params"]
    model_config = config["model_params"]

    frame_path = os.path.join(base_path, latent_config["frame_path"])
    model_path = os.path.join(base_path, latent_config["model_path"])
    is_model_pth = latent_config["is_model_pth"]
    device = latent_config["device"]
    img_dim = (latent_config["img_dim_H"], latent_config["img_dim_W"])
    downsample_factor = latent_config["downsample_factor"]
    patchSize_list = get_patchSize_list(img_dim, downsample_factor, log=True)
    save_latent_path = os.path.join(base_path, latent_config["save_latent_path"])
    save_latent_folder_name = latent_config["save_latent_folder_name"]
    is_upsample = latent_config["is_upsample"]

    factor = latent_config["factor"]
    num_frames = latent_config["num_frames"]
    img_format = latent_config["img_format"]
    tensors_png_path = os.path.join(base_path, latent_config["tensors_png_path"])
    tensors_png_folder_name = latent_config["tensors_png_folder_name"]
    vae_model = VAE(dataLoader_config["img_channels"], model_config)

    print("\n----- Loading the model -----\n")
    ## Load the model from the model_path

    model, _, _, _, _, _, _ = loadModel(
        model_path,
        model_class=vae_model,
        discriminator_class=Discriminator(im_channels=3),
        pth=is_model_pth,
        device=device,
    )

    model.eval()
    print("\n----- Model loaded successfully, set in eval() mode -----\n")

    print("\n----- Saving raw latents from the frames -----\n")
    # Save latents from the frames
    raw_latents_path = save_raw_latent_all(
        frame_path,
        model,
        save_latent_path,
        patchSize_list,
        save_latent_folder_name,
        num_frames,
        is_upsample,
        factor,
    )

    print("\n------ Converting the latents to an image format -----\n")

    save_tensor_as_png_all(
        raw_latents_path,
        tensors_png_path,
        tensors_png_folder_name,
        img_format,
    )
    print("\n------ Latents converted to image format successfully -----\n")


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
    save_latents(args)
