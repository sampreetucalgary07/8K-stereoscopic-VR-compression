import os
import torch
import torch.optim as optim
from src.model_func import loadModel


def model_size(model_path, print_value=True, return_value=False):
    if print_value:
        print(f"Model Size : {os.path.getsize(model_path) / (1024 * 1024)} MB")
    if return_value:
        return os.path.getsize(model_path) / (1024 * 1024)


def convertModel(model_path, save_path, model, device="cuda"):
    # function to convert a .pth model to .pt model

    pt_model, _, _, _, _ = loadModel(model_path, model, pth=True)
    pt_model = pt_model.to(device)
    pt_model.eval()
    torch.save(pt_model.state_dict(), save_path)
    print("New model saved at: ", save_path)
