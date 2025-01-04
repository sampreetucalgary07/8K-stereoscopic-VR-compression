from tqdm import tqdm
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys

sys.path.append(os.path.dirname(__file__))

# local imports
from utils.file_utils import make_folder, filenames
from utils.tensor_utils import (
    normalize_tensor,
    image_to_tensor,
    tensor_to_image,
    path_to_imagelist,
)


def create_diff_tensor(image1, image2, is_tensor=False, normalize=False, device="cuda"):
    # Convert images to tensors
    if is_tensor:
        image1_tensor = image1.to(device)
        image2_tensor = image2.to(device)

    else:
        image1_tensor = image_to_tensor(image1).to(device)
        image2_tensor = image_to_tensor(image2).to(device)
    # Calculate pixel difference percentage
    diff_tensor = torch.sub(image1_tensor, image2_tensor)
    if normalize == False:
        return diff_tensor
    else:
        diff_tensor, max_value, min_value = normalize_tensor(diff_tensor)
        return diff_tensor, max_value, min_value


def save_diff_tensors(
    imagelist_path1,
    imagelist_path2,
    destination_path=None,
    permute_values=[0, 1, 3, 2],
):

    foldername, _, base_path = filenames(imagelist_path1)

    if destination_path == None:
        destination_path = base_path

    dest_folder = make_folder(
        folder_name=foldername + "_diff",
        folder_path=destination_path,
        remove_if_exists=True,
    )

    imageList1 = path_to_imagelist(imagelist_path1)
    imageList2 = path_to_imagelist(imagelist_path2)

    if len(imageList1) != len(imageList2):
        print("Image lists are not of the same length")
        return

    for image1, image2 in zip(imageList1, tqdm(imageList2, total=len(imageList2))):
        diff_tensor, max_value_tensor, min_value_tensor = create_diff_tensor(
            image1, image2, normalize=True
        )
        # print(diff_tensor.shape)
        file_name, ext, _ = filenames(image1)
        new_file_name = file_name.replace("out_", "diff_") + ext
        save_file_path = os.path.join(dest_folder, new_file_name)
        # print(save_file_path)
        # save max and min values in a csv file
        with open(os.path.join(dest_folder, "max_min_values.csv"), "a") as file:
            file.write(f"{new_file_name},{max_value_tensor},{min_value_tensor}\n")

        tensor_to_image(diff_tensor, save_file_path, permute_values)

    print("Diff images saved to: ", dest_folder)
    print(
        "Max and Min values saved to: ", os.path.join(dest_folder, "max_min_values.csv")
    )
    return dest_folder


def add_min_max_values(tensor, max_value, min_value):
    return torch.add(torch.mul(tensor, max_value - min_value), min_value)


def sub_diff_tensor(diff_tensor, image, is_image_tensor=False, device="cuda"):
    # Convert images to tensors
    if is_image_tensor:
        image_tensor = image.to(device)
    else:
        image_tensor = image_to_tensor(image).to(device)
    # Add the difference tensor to the image tensor
    new_image_tensor = torch.sub(image_tensor, diff_tensor)
    return new_image_tensor
