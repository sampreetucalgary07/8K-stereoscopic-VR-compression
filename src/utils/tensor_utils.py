import os
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from PIL import Image

##################################### Tensor preprocessing #####################################


def image_to_tensor(
    image_path: str,
    normalize: bool = True,
    device: str = "cuda",
):
    if normalize:
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    else:
        transform = [
            transforms.ToTensor(),
        ]
    image = Image.open(image_path)
    tensor = torchvision.transforms.Compose(transform)(image).unsqueeze(0).to(device)
    return tensor


def tensor_to_image(tensor, save_path=None, permute_values=[0, 1, 3, 2]):
    if save_path:
        torchvision.utils.save_image(tensor.permute(permute_values), save_path)
    else:
        return tensor.detach().cpu().permute(permute_values).numpy()


def path_to_imagelist(image_list_path, start_num=0, end_num=None):
    image_list = sorted(os.listdir(image_list_path))[start_num:end_num]
    images_list = [os.path.join(image_list_path, img) for img in image_list]

    return images_list


def normalize_tensor(tensor, transform=transforms.Compose([transforms.ToTensor()])):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor, max_val, min_val


def show_tensor_image(
    tensor, permute_values=[1, 2, 0], fig_size=(20, 20), show_cmap=False
):
    plt.figure(figsize=fig_size)
    if show_cmap:
        plt.imshow(tensor.squeeze(0).permute(permute_values).cpu().numpy())
        plt.colorbar()
    else:
        plt.imshow(tensor.squeeze(0).permute(permute_values).cpu().numpy())
    plt.show()


def num_zeros(tensor, print_values=False):
    # Count the number of zeros in the tensor
    num_zeros = torch.sum(torch.eq(tensor, 0)).item()
    if print_values:
        print(f"Number of zeros: {num_zeros}")
    return num_zeros


def add_min_max_values(tensor, max_value, min_value):
    return torch.add(torch.mul(tensor, max_value - min_value), min_value)
