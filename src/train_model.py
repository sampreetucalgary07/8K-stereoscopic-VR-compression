import torch
import torch.optim as optim
import os
import time
import numpy as np
from tqdm import tqdm

##User defined libraries
from model.discriminator import Discriminator
from src.utils.log_utils import append_to_csv
from src.model_func import SaveBestModel, EarlyStopping


def train_vae(
    model,
    dataset,
    epoch_start_end: tuple,
    optimizer_g,
    optimizer_d,
    discriminator,
    recon_criterion,
    lpips_model,
    disc_criterion,
    training_config,
    save_model_path: str,
    save_model_name: str,
    save_model_as_pth: bool,
    save_csv_path: str,
    retrain_best_loss: float = float("inf"),
    tl_loss = None,
):
    acc_steps = training_config["acc_steps"]
    disc_step_start = training_config["disc_start"]
    save_model_start = training_config["save_model_start"]
    gen_lr = training_config["gen_lr"]
    disc_lr = training_config["disc_lr"]
    device = training_config["device"]

    save_model = SaveBestModel(
        save_model_path,
        save_model_name,
        best_valid_loss=retrain_best_loss,
        remove_previous_model=True,
    )
    early_stopping = EarlyStopping(tolerance=3)

    """
    Function to train the VAE model
    """
    print("\n ----- TRAINING CONFIGURATION -------\n")
    print("Epoch start and end: ", epoch_start_end)
    print("Discriminator start step: ", disc_step_start)
    print("Accumulation steps: ", acc_steps)
    print("Save model start epoch: ", save_model_start)
    print("Device: ", device)
    print("Learning rate (gen): ", gen_lr)
    print("Learning rate (disc): ", disc_lr)
    print("Save path: ", save_model_path)
    print("Save csv path: ", save_csv_path)
    print("Save model name: ", save_model_name)
    print("Save model as pth: ", save_model_as_pth)
    print("\n ------------------------------------------------------ \n")

    print("\n ----- TRAINING START ----- \n")
    model.train()
    step_count = 0
    for epoch_idx in range(epoch_start_end[0], epoch_start_end[1]):
        start_time = time.time()
        recon_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []

        optimizer_g.zero_grad()  # type: ignore
        optimizer_d.zero_grad()  # type: ignore
        # print("Epoch: ", epoch_idx + 1)
        for im, _ in tqdm(
            dataset, desc="Epoch : " + str(epoch_idx + 1), total=len(dataset)
        ):
            step_count += 1
            im = im.float().to(device)

            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z = model_output
            # print(z.shape)

            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im)
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = recon_loss
            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(
                    disc_fake_pred,
                    torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
                )
                gen_losses.append(
                    training_config["disc_weight"] * disc_fake_loss.item()
                )
                g_loss += training_config["disc_weight"] * disc_fake_loss / acc_steps
            # Perceptual Loss
            lpips_loss = torch.mean(lpips_model(output, im)) / acc_steps
            perceptual_losses.append(
                training_config["perceptual_weight"] * lpips_loss.item()
            )
            g_loss += training_config["perceptual_weight"] * lpips_loss / acc_steps
            losses.append(g_loss.item())
            g_loss.backward()

            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(
                    disc_fake_pred,
                    torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device),
                )
                disc_real_loss = disc_criterion(
                    disc_real_pred,
                    torch.ones(disc_real_pred.shape, device=disc_real_pred.device),
                )
                disc_loss = (
                    training_config["disc_weight"]
                    * (disc_fake_loss + disc_real_loss)
                    / 2
                )
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()

            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()

        print(
            "\nFinished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | "
            #'Codebook : {:.4f} '
            "| G Loss : {:.4f} | D Loss {:.4f}".format(
                epoch_idx + 1,
                np.mean(recon_losses),
                np.mean(perceptual_losses),
                np.mean(gen_losses),
                np.mean(disc_losses),
            )
        )
        data_for_csv = {
            "epoch": epoch_idx + 1,
            "recon_loss": np.mean(recon_losses),
            "perceptual_loss": np.mean(perceptual_losses),
            "disc_loss": np.mean(disc_losses),
            "gen_loss": np.mean(gen_losses),
            "time_taken": time.time() - start_time,
        }
        ## adding data to csv file
        append_to_csv(save_csv_path, data_for_csv)
        
        if tl_loss is not None:
            if np.mean(perceptual_losses) <= tl_loss:
                print("\nTransfer learning loss achieved. Stopped training..")
                break
        

        if early_stopping.save(np.mean(recon_losses)):
            print("\nEarly Stopping at epoch: ", epoch_idx + 1)
            break

        # model Saving
        if training_config["save_model"]:
            _ = None
            if epoch_idx + 1 >= save_model_start:
                save_model.save(
                    np.mean(recon_losses),
                    epoch_idx + 1,
                    model,
                    optimizer_g,
                    optimizer_d,
                    discriminator,
                    [recon_criterion, lpips_loss, disc_criterion],
                    save_pth=save_model_as_pth,
                )
        # print("\n")
        print("---------------------------------------------------\n")

    print("\n ----- TRAINING FINISHED ----- \n")
    print("Logs saved at: ", save_csv_path)
