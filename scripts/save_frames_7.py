"""This script is used to save frames from the given latents using the model"""

### importing standard libraries
import os
import yaml
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

### importing user defined libraries

from src.model_to_latents import reload_tensor_from_png
from src.model_func import loadModel
from model.vae_model import VAE
from src.latent_to_frames import save_raw_frame_all
from src.patches import get_patchSize_list
from model.discriminator import Discriminator


def save_frames(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]
    dataLoader_config = config["dataLoader_params"]
    model_config = config["model_params"]
    reconstr_config = config["reconstruction_params"]
    latents_as_img_path = os.path.join(
        base_path, reconstr_config["latents_as_img_path"]
    )
    latent_min_max_csv_path = os.path.join(
        base_path, reconstr_config["latent_min_max_csv_path"]
    )
    output_folder = os.path.join(base_path, reconstr_config["output_folder"])
    save_folder_name_raw_tensors = reconstr_config["save_folder_name_raw_tensors"]
    save_folder_name_raw_frames = reconstr_config["save_folder_name_raw_frames"]
    img_format = reconstr_config["img_format"]
    num_frames = reconstr_config["num_frames"]
    model_path = os.path.join(base_path, reconstr_config["model_path"])
    is_model_pth = reconstr_config["is_model_pth"]
    device = reconstr_config["device"]
    vae_model = VAE(dataLoader_config["img_channels"], model_config)

    upsample_factor = reconstr_config["upsample_factor"]
    latent_dim = (reconstr_config["latent_dim_H"], reconstr_config["latent_dim_W"])
    patchSize_list = get_patchSize_list(latent_dim, upsample_factor, log=True)
    factor = reconstr_config["factor"]

    print("\n----- Reloading tensors from images -----\n")

    recon_latents_path = reload_tensor_from_png(
        inp_latent_png_path=latents_as_img_path,
        save_path=output_folder,
        save_folder_name=save_folder_name_raw_tensors,
        csv_file_path=latent_min_max_csv_path,
        img_format=img_format,
        return_save_path=True,
    )

    # print(recon_latents_path)
    print("\n----- Tensors saved to output_folder -----\n")

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
    print(recon_latents_path)
    print("\n----- Saving frames from the latents -----\n")
    save_raw_frame_all(
        recon_latents_path,
        model,
        output_folder,
        save_folder_name_raw_frames,
        patchSize_list,
        num_frames,
        True,
        factor,
        img_format,
        device,
    )
    print("\n----- Frames saved successfully -----\n")


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
    save_frames(args)
