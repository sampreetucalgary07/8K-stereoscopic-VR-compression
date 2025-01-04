### importing standard libraries
import yaml
import torch
import shutil
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
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
from src.utils.log_utils import init_csv, csv_dir_check, append_to_csv
from src.utils.file_utils import check_image_size
from src.model_func import SaveBestModel, loadModel
from src.train_model import train_vae


def transfer_learning(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)

    # Extract the config parameters #
    base_path = config["default_params"]["base_path"]
    print(base_path)
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

    base_model_path = os.path.join(base_path, config["tl_params"]["base_model_path"])
    copy_model = config["tl_params"]["copy_model"]
    model_copy_path = os.path.join(base_path, config["tl_params"]["model_copy_path"])
    # copy the base model to the model_copy_path
    if copy_model:
        os.makedirs(os.path.dirname(model_copy_path), exist_ok=True)
        shutil.copy(base_model_path, model_copy_path)
        print("\n ----- Model copied successfully ----- \n")
        print("\n ----- Model copied to: ", model_copy_path, " ----- \n")
    else:
        model_copy_path = base_model_path
        print("\n ----- Model not copied ----- \n")

    is_model_pth = config["tl_params"]["is_model_pth"]
    epoch = config["tl_params"]["epoch"]
    enc_unfreeze_perc = config["tl_params"]["enc_unfreeze_perc"]
    dec_unfreeze_perc = config["tl_params"]["dec_unfreeze_perc"]
    save_model_path = os.path.join(base_path, config["tl_params"]["save_model_path"])
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path, exist_ok=True)
    save_model_name = config["tl_params"]["save_model_name"]
    save_model_as_pth = config["tl_params"]["save_model_as_pth"]
    save_csv_path = os.path.join(base_path, config["tl_params"]["save_csv_path"])
    overwrite_prev_csv = config["tl_params"]["overwrite_prev_csv"]
    prev_best_loss = config["tl_params"]["prev_best_loss"]

    csv_dir_check(save_csv_path, overwrite_prev_csv)
    fieldnames = [
        "epoch",
        "recon_loss",
        "perceptual_loss",
        "disc_loss",
        "gen_loss",
        "time_taken",
    ]

    init_csv(save_csv_path, fieldnames)

    print("\n ----- Transfer Learning Configuration -------\n")
    print("Base model path: ", base_model_path)
    print("Copy model: ", copy_model)
    print("Model copy path: ", model_copy_path)
    print("Epoch: ", epoch)
    print("Is model pth: ", is_model_pth)
    print("Device: ", device)
    print("Encoder trainable layers percentage: ", enc_unfreeze_perc * 100)
    print("Decoder trainable layers percentage: ", dec_unfreeze_perc * 100)
    print("Save csv path: ", save_csv_path)

    print("\n ------------------------------------------------------ \n")

    (
        tl_model,
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
    print(f"Model loaded successfully")

    tl_model.eval()
    
    print((f"Model set to device: {device}"))
    print("\n ------------------------------------------------------ \n")

    model_module_list = list(tl_model.modules())
    model_param_list = list(tl_model.parameters())

    divide_layer = tl_model.post_quant_conv

    for idx, (module, param) in enumerate(zip(model_module_list[1:], model_param_list)):
        if module == divide_layer:
            print("Encoder last layer idx : ", idx, "Module : ", module, "\n")
            divide_idx = idx
            break

    encoder_modules = model_module_list[1:][: divide_idx + 1]
    decoder_modules = model_module_list[1:][divide_idx + 1 :]

    encoder_params = model_param_list[: divide_idx + 1]
    decoder_params = model_param_list[divide_idx + 1 :]

    print("\nNo. of encoder layers: ", len(encoder_modules))
    print("\nNo. of decoder layers: ", len(decoder_modules))

    print("\n ---- Unfreezing all the model layers... ---- \n")

    for mod in model_param_list[:]:
        mod.requires_grad = True

    print("\n ---- Freezing the encoder layers ---- \n")
    for idx, enc_param in enumerate(encoder_params):
        if idx < int((1 - enc_unfreeze_perc) * len(encoder_params)):
            enc_param.requires_grad = False

    print("\n ---- Freezing the decoder layers ---- \n")
    for idx, dec_param in enumerate(decoder_params):
        if idx < int((1 - dec_unfreeze_perc) * len(decoder_params)):
            dec_param.requires_grad = False

    no_of_train_layers = 0
    total_layers = 0

    for mod in tl_model.parameters():
        total_layers += 1
        # print(mod.requires_grad)
        if mod.requires_grad:
            no_of_train_layers += 1

    print("\n ----- Transfer Learning Summary -------------- \n")
    print("\nTotal layers: ", total_layers)
    print("\nTrainable layers: ", no_of_train_layers)
    print("\nFrozen layers: ", total_layers - no_of_train_layers)
    print("\nModel path: ", model_copy_path)
    print("\n ------------------------------------------------------ \n")

    recon_criterion = torch.nn.MSELoss()  # L1/L2 loss for Reconstruction
    disc_criterion = torch.nn.BCEWithLogitsLoss()  # Disc Loss can even be BCEWithLogits
    lpips_model = LPIPS().eval().to(device)
    # discriminator = Discriminator(im_channels=3).to(device)

    train_vae(
        tl_model,
        dataset,
        (epoch, epoch + config["tl_params"]["epoch"]),
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
        retrain_best_loss=float("inf"),
        tl_loss = prev_best_loss,
    )


if __name__ == "__main__":
    base_dir = "/home/ubuntu/personal/VAE_Analysis"
    parser = argparse.ArgumentParser(description="Arguments for vae training")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(base_dir, "configs", "E1_config.yaml"),
        help="Path to the config file",
    )
    args = parser.parse_args()
    transfer_learning(args)
