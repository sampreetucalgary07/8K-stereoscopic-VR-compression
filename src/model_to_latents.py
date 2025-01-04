""" This script generates latent tensors from a dataset using a model"""

### importing standard libraries
from tqdm import tqdm
import os
import torch
import dask
from dask.diagnostics import ProgressBar  # type: ignore
import csv
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision


### importing user defined libraries
from src.patches import frame_from_patches
from src.utils.file_utils import filenames, make_folder
from src.utils.tensor_utils import path_to_imagelist
from src.utils.tensor_utils import (
    tensor_to_image,
    normalize_tensor,
    image_to_tensor,
    add_min_max_values,
)
from src.dataLoader import createTorchDataset


### class to predict latent tensors from a frame using the model given
class frame_to_latent(frame_from_patches):
    def __init__(
        self,
        image,
        patch_size_list,
        model,
        upsample=True,
        factor=1,
        is_tensor=False,
        device="cuda",
    ):
        super().__init__(
            image=image,
            patch_size_list=patch_size_list,
            model=model,
            upsample=upsample,
            factor=factor,
            is_tensor=is_tensor,
            device=device,
        )

    def preprocess_patch(self, patch):
        # patch = (2 * patch) - 1
        patch = patch.unsqueeze(0)
        patch = transforms.Normalize((0.5,), (0.5,))(patch)

        return patch

    def get_model_output_patch(self, patch):
        encoded_output, _ = self.model.encode(patch.to(self.device))
        # encoded_output = torch.clamp(encoded_output, -1.0, 1.0)
        # encoded_output = (encoded_output + 1) / 2
        # print(encoded_output.shape)
        return encoded_output


# @dask.delayed
def save_raw_latent(
    img,
    img_path,
    model,
    dest_folder,
    patch_list,
    upsample=False,
    factor=1,
    is_tensor=False,
):
    encoded_tensor = frame_to_latent(
        img,
        patch_list,
        model,
        upsample=upsample,
        factor=factor,
        is_tensor=is_tensor,
    ).get_result()
    # remove .jpg from base_name

    save_path = os.path.join(
        dest_folder, "enc_" + os.path.splitext(os.path.basename(img_path))[0] + ".pt"
    )

    # tensor_to_image(encoded_image, save_path, permute_values=[0, 1, 2, 3])
    torch.save(encoded_tensor, save_path)


def save_raw_latent_all(
    input_folder,
    model,
    destination_path,
    patch_list,
    output_folder_name=None,
    num_images: list = [None, None],
    upsample=False,
    factor=1,
):
    foldername, _, base_path = filenames(input_folder)

    if destination_path == None:
        destination_path = base_path

    if output_folder_name:
        foldername = output_folder_name
    else:
        foldername = foldername + "_latent_tensors"

    dest_folder = make_folder(
        foldername,
        folder_path=destination_path,
        remove_if_exists=True,
    )

    sample_dataset = createTorchDataset(
        input_folder,
        start_num_image=num_images[0],
        end_num_image=num_images[1],
        transforms=[transforms.ToTensor()],
        channel_last=False,
    )

    dataset = DataLoader(
        sample_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
    )

    # imageList = path_to_imagelist(input_folder)[num_images[0] : num_images[1]]
    print("\n")
    for img_tensor, img_path in tqdm(
        dataset, total=len(sample_dataset), desc="Saving latent tensors ..."
    ):
        # for img in tqdm(imageList, total=len(imageList), desc="Saving latent tensors ..."):
        # print(img_path[0])
        save_raw_latent(
            img_tensor,
            img_path[0],
            model,
            dest_folder,
            patch_list,
            upsample=upsample,
            factor=factor,
            is_tensor=True,
        )

    print("\n")

    print("\nLatents saved to: ", dest_folder)
    print("\n----- Latents saved successfully -----\n   ")
    return dest_folder


# @dask.delayed  # type: ignore
def save_tensor_as_png(
    tensor,
    save_tensor_path,
    csv_file_path,
    img_format=".png",
    extra_pixel=None,
    csv_status="w",
):

    tensor, max_value, min_value = normalize_tensor(tensor)
    # print(f"max val: {max_value}, min val: {min_value}")
    # print(f"torch max val: {torch.max(tensor)}, torch min val: {torch.min(tensor)}")
    if extra_pixel is not None:
        tensor = torch.cat([tensor, extra_pixel], dim=2)

    with open(csv_file_path, csv_status, newline="\n") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                os.path.basename(save_tensor_path),
                max_value.item(),
                min_value.item(),
            ]
        )

    tensor_to_image(
        tensor,
        save_tensor_path,
        permute_values=[0, 1, 2, 3],
    )


def save_tensor_as_png_all(
    input_tensors_path,
    save_path=None,
    save_folder_name="tensors_png",
    img_format=".png",
    save_residual_ext: str = None, # type: ignore
    extra_pixel=None,
    return_values=False,
    num_images: list = [None, None],
):

    if save_residual_ext is not None:
        if save_residual_ext == "L":
            ext_1 = "L.pt"
            ext_2 = "R.pt"
        elif save_residual_ext == "T":
            ext_1 = "T.pt"
            ext_2 = "B.pt"
        else:
            raise ValueError(
                "save_residual_ext should be either 'L' or 'T' as a string"
            )
    else:
        ext_1 = ".pt"
        ext_2 = ".pt"
    _, _, baseFolder = filenames(input_tensors_path)

    if save_path is not None:
        baseFolder = save_path

    save_path = make_folder(save_folder_name, baseFolder, remove_if_exists=True)

    csv_file_path = os.path.join(save_path, "min_max_values.csv")
    if extra_pixel is not None:
        extra_pixel = torch.zeros(1, 3, 1, 240).to("cuda")
    # delayed_tasks = []  # list to store all delayed tasks

    file_no = 0
    for tensor_file in tqdm(
        sorted(os.listdir(input_tensors_path))[num_images[0] : num_images[1]], desc="Saving latents as frames ..."
    ):
        csv_status = "w" if file_no == 0 else "a"
        if tensor_file.endswith(ext_1):
            tensor_ext1_path = os.path.join(input_tensors_path, tensor_file)
            tensor_ext1 = torch.load(tensor_ext1_path)
            save_tensor_path = os.path.join(
                save_path, tensor_file.replace(".pt", img_format)
            )
            # delayed_tasks.append(
            save_tensor_as_png(
                tensor_ext1,
                save_tensor_path,
                csv_file_path,
                img_format,
                extra_pixel,
                csv_status,
            )
            # )
            file_no += 1

            tensor_ext2_path = tensor_ext1_path.replace(ext_1, ext_2)
            tensor_file_2 = tensor_file.replace(ext_1, ext_2)
            if os.path.exists(tensor_ext2_path):
                tensor_ext2 = torch.load(tensor_ext2_path)

                if save_residual_ext is None:
                    save_tensor_path = os.path.join(
                        save_path, tensor_file_2.replace(".pt", img_format)
                    )
                    # delayed_tasks.append(
                    save_tensor_as_png(
                        tensor_ext2,
                        save_tensor_path,
                        csv_file_path,
                        img_format,
                        extra_pixel,
                        csv_status,
                    )
                else:
                    save_tensor_path = os.path.join(
                        save_path, tensor_file_2.replace(".pt", "_diff" + img_format)
                    )
                    save_tensor_as_png(
                        torch.sub(tensor_ext1, tensor_ext2),
                        save_tensor_path,
                        csv_file_path,
                        img_format,
                        extra_pixel,
                        csv_status,
                    )
                    # )
                file_no += 1
    print("\n")
    # print("Computing all delayed tasks ...")
    # with ProgressBar():
    #     dask.compute(*delayed_tasks)  # type: ignore

    print(f"All tensors saved in {save_path}")
    if return_values:
        return save_path


def reload_tensor_from_png(
    inp_latent_png_path,
    save_path=None,
    save_folder_name="tensors_from_img",
    csv_file_path=None,
    img_format=".png",
    return_save_path=False,
    extra_pixel=None,
):
    _, _, baseFolder = filenames(inp_latent_png_path)

    if save_path is not None:
        baseFolder = save_path

    save_path = make_folder(save_folder_name, baseFolder, remove_if_exists=True)

    if csv_file_path is None:
        csv_file = os.path.join(inp_latent_png_path, "min_max_values.csv")
    else:
        csv_file = csv_file_path

    with open(csv_file, "r", newline="\n") as csvfile:
        csvreader = csv.reader(csvfile)
        for img in tqdm(
            sorted(os.listdir(inp_latent_png_path)),
            total=len(os.listdir(inp_latent_png_path)),
            desc="Reloading tensors from images ...",
        ):
            if img.endswith(img_format):

                for row in csvreader:
                    if row[0] == img:
                        img_path = os.path.join(inp_latent_png_path, img)
                        tensor = image_to_tensor(img_path, normalize=False)
                        # print("type of tensor", type(tensor))
                        # print("max val _ 1", torch.max(tensor))
                        # print("min val _ 1 ", torch.min(tensor))
                        max_value = float(row[1])
                        min_value = float(row[2])
                        # print(f"max val: {max_value}, min val: {min_value}")
                        tensor = add_min_max_values(tensor, max_value, min_value)
                        if extra_pixel is not None:
                            tensor = torch.cat([tensor, extra_pixel], dim=2)
                        # print(tensor.shape)

                        torch.save(
                            tensor,
                            os.path.join(save_path, img.replace(img_format, ".pt")),
                        )
                        break

    print(f"\nAll transformed pngs saved in {save_path}")
    if return_save_path:
        return save_path
