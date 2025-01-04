# importing standard libraries
from tqdm import tqdm
import os
import torch
import shutil
import re
import sys
import dask
from dask.diagnostics import ProgressBar  # type: ignore
import glob
from torchvision import transforms
from torch.utils.data import DataLoader


# dask.config.set(scheduler="processes")

# import glob
sys.path.append(os.path.dirname(__file__))

# imoporting custom libraries
from utils.file_utils import filenames, make_folder, check_image_size
from utils.patch_utils import get_patchSize_list, get_sub_image
from utils.tensor_utils import tensor_to_image, image_to_tensor, path_to_imagelist
from src.dataLoader import createTorchDataset
from src.metrics import calculate_ssim


# @dask.delayed  # type: ignore
def save_sub_image(img_tensor, img_path, key, value, dest_folder, is_tensor, device):
    """Function to save the sub image."""
    sub_img = get_sub_image(img_tensor, value, False, is_tensor, device).unsqueeze(0)
    save_path = os.path.join(dest_folder, key + "_" + os.path.basename(img_path))
    # sub_img.save(save_path)
    tensor_to_image(sub_img, save_path, permute_values=[0, 1, 2, 3])


def frame_to_patches(
    input_folder,
    img_dim,
    downsample_factor,
    destination_path=None,
    output_folder_name=None,
    num_images: list = [None, None],
    return_dest_path=False,
    device="cuda",
):
    """This function creates patches from a folder containing images."""
    foldername, _, base_path = filenames(input_folder)

    if destination_path == None:
        destination_path = base_path

    if output_folder_name:
        foldername = output_folder_name
    else:
        foldername = foldername + "_patches"

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
        num_workers=16,
    )
    # imageList = path_to_imagelist(input_folder)[num_images[0] : num_images[1]]
    patch_list = get_patchSize_list(img_dim, downsample_factor)
    # print(patch_list)
    for img_tensor, img_path in tqdm(
        dataset, total=len(sample_dataset), desc="Generating Patches:"
    ):
        # img_tensor = image_to_tensor(img)
        tasks = []
        for key, value in patch_list.items():
            # print(key, value[0], value[1] - 1, value[2], value[3] - 1)
            # tasks.append(save_sub_image(img, key, value, dest_folder, True, device))
            save_sub_image(
                img_tensor, img_path[0], key, value, dest_folder, True, device
            )
            # print(sub_img.shape)
            # print(save_path)
            # tasks.append(save_sub_image(sub_img, save_path))

        # dask.compute(*tasks)  # type: ignore

    print("Patches saved to: ", dest_folder)
    if return_dest_path:
        return dest_folder


class filter_patches:
    """Class to filter patches based on a condition.
    The condition is defined in the if_condition method.
    See examples in the subclasses."""

    def __init__(
        self,
        input_folder_path: str,
        num_patches: int,
        patch_format: str,
        dest_path: str = "",
        folder_name=None,
    ):
        self.input_folder_path = input_folder_path
        self.num_patches = num_patches
        self.dest_path = dest_path
        self.folder_name = folder_name
        self.patch_format = patch_format

    def folder(self):
        foldername, _, base_path = filenames(self.input_folder_path)

        if self.dest_path == "":
            self.dest_path = base_path

        if self.folder_name:
            foldername = self.folder_name

        dest_folder_path = make_folder(
            folder_name=foldername,
            folder_path=self.dest_path,
            remove_if_exists=True,
        )
        return dest_folder_path

    def if_condition(self, ref_frame, frame, frame_no):
        raise NotImplementedError

    def if_condition_inter(self, ref_patch, patch, frame_no):
        raise NotImplementedError

    @dask.delayed  # type: ignore
    def if_condition_copy(self, ref_frame, frame, frame_no, dest_path):
        if self.if_condition(ref_frame, frame, frame_no):
            shutil.copy(frame, dest_path)

    @dask.delayed  # type: ignore
    def if_condition_inter_copy(self, ref_patch, patch, patch_no, dest_path):
        if self.if_condition_inter(ref_patch, patch, patch_no):
            shutil.copy(patch, dest_path)

    def patch_filter(self):
        self.dest_path = self.folder()
        # all_patches = os.listdir(self.input_folder_path)
        for i in tqdm(
            range(1, self.num_patches + 1),
            total=self.num_patches,
            desc="Filtering patches...",
        ):
            pattern = f"Patch_{i}_*.*"
            filtered_files = sorted(
                glob.glob(os.path.join(self.input_folder_path, pattern))
            )
            ref_frame = filtered_files[0]
            shutil.copy(ref_frame, self.dest_path)
            delayed_tasks = []
            for frame_no, frame in enumerate(filtered_files[1:]):
                # print(self.if_cond(ref_frame, frame, frame_no))
                delayed_tasks.append(
                    self.if_condition_copy(ref_frame, frame, frame_no, self.dest_path)
                )

            dask.compute(*delayed_tasks)  # type: ignore

    def inter_patch_filter(self):
        self.inter_patch_dest_path = make_folder(
            folder_name="inter_patch_filter",
            folder_path=self.dest_path,
            remove_if_exists=True,
        )
        all_patches = sorted(os.listdir(self.dest_path))
        ref_patch = os.path.join(self.dest_path, all_patches[0])
        shutil.copy(ref_patch, self.inter_patch_dest_path)
        delayed_tasks = []
        for patch_no, patch in tqdm(
            enumerate(all_patches[1:]),
            total=len(all_patches[1:]),
            desc="\nFiltering patches internally",
        ):
            if patch.endswith(self.patch_format):

                delayed_tasks.append(
                    self.if_condition_inter_copy(
                        ref_patch,
                        os.path.join(self.dest_path, patch),
                        patch_no,
                        self.inter_patch_dest_path,
                    )
                )

        with ProgressBar():
            dask.compute(*delayed_tasks)  # type: ignore

    def get_dest_path_info(self):
        check_image_size(self.dest_path)
        print(f"Destination path: {self.dest_path}")


class filter_patches_ssim(filter_patches):
    def __init__(
        self,
        input_folder_path,
        num_patches,
        patch_format,
        dest_path,
        folder_name,
        threshold=1.0,
        inter_threshold=1.0,
    ):
        super().__init__(
            input_folder_path, num_patches, patch_format, dest_path, folder_name
        )
        self.threshold = threshold
        self.inter_threshold = inter_threshold

    def if_condition(self, ref_frame, frame, frame_no):
        return calculate_ssim(ref_frame, frame).get_result() < self.threshold

    def if_condition_inter(self, ref_patch, patch, patch_no):
        return calculate_ssim(ref_patch, patch).get_result() < self.inter_threshold


class filter_patches_sample(filter_patches):
    def __init__(
        self,
        input_folder_path,
        num_patches,
        patch_format,
        dest_path,
        folder_name=None,
        sample_factor=5,
        inter_sample_factor=5,
    ):
        super().__init__(
            input_folder_path, num_patches, patch_format, dest_path, folder_name
        )
        self.sample_factor = sample_factor
        self.inter_sample_factor = inter_sample_factor

    def if_condition(self, ref_frame, frame, frame_no):
        return frame_no % self.sample_factor == 0

    def if_condition_inter(self, ref_patch, patch, patch_no):
        return patch_no % self.sample_factor == 0


class filter_patches_ssim_sample(filter_patches):
    def __init__(
        self,
        input_folder_path,
        num_patches,
        patch_format,
        dest_path,
        folder_name=None,
        threshold=1,
        sample_factor=5,
        inter_threshold=1.0,
        inter_sample_factor=5,
    ):
        super().__init__(
            input_folder_path, num_patches, patch_format, dest_path, folder_name
        )
        self.sample_factor = sample_factor
        self.threshold = threshold
        self.inter_threshold = inter_threshold
        self.inter_sample_factor = inter_sample_factor

    def if_condition(self, ref_frame, frame, frame_no):
        return (
            frame_no % self.sample_factor == 0
            and calculate_ssim(ref_frame, frame).get_result() < self.threshold
        )

    def if_condition_inter(self, ref_patch, patch, patch_no):
        return (
            patch_no % self.inter_sample_factor == 0
            and calculate_ssim(ref_patch, patch).get_result() < self.inter_threshold
        )


############################################################################################################
class frame_from_patches:
    """Class to get a frame from patches from a model.
    Args:
    image : torch.tensor : (1, C, H, W)
    patch_size_list : dict : {'Patch_1': [0, 64, 0, 64]}
    model : torch.nn.Module : model to be used
    upsample : bool : (upsample the image)
    factor : int : (upsampling factor)
    is_tensor : bool : (if the image is a tensor)
    device : str : ('cuda' or 'cpu')
    """

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
        self.is_tensor = is_tensor
        self.image = image
        if self.is_tensor:
            self.test_tensor = self.image
        else:
            self.test_tensor = image_to_tensor(self.image)
        self.patch_size_list = patch_size_list
        self.model = model
        self.device = device
        self.factor = factor
        self.upsample = upsample
        if self.upsample:
            self.zero_tensor = torch.zeros(
                int(self.test_tensor.shape[0]),
                int(self.test_tensor.shape[1]),
                int(self.test_tensor.shape[2] * self.factor),
                int(self.test_tensor.shape[3] * self.factor),
            ).to(self.device)
        else:
            self.zero_tensor = torch.zeros(
                int(self.test_tensor.shape[0]),
                int(self.test_tensor.shape[1]),
                int(self.test_tensor.shape[2] / self.factor),
                int(self.test_tensor.shape[3] / self.factor),
            ).to(self.device)

    def preprocess_patch(self, patch):
        raise NotImplementedError

    def get_model_output_patch(self, patch):
        raise NotImplementedError

    def get_result(self):
        with torch.no_grad():
            for _, value in self.patch_size_list.items():
                patch = get_sub_image(
                    self.test_tensor, value, print_size=False, is_tensor=True
                )
                patch_preprocess = self.preprocess_patch(patch)
                result_patch = self.get_model_output_patch(patch_preprocess)
                if self.upsample:
                    self.zero_tensor[0][
                        :,
                        int(value[0] * self.factor) : int(value[1] * self.factor),
                        int(value[2] * self.factor) : int(value[3] * self.factor),
                    ] = result_patch[0]
                else:
                    self.zero_tensor[0][
                        :,
                        int(value[0] / self.factor) : int(value[1] / self.factor),
                        int(value[2] / self.factor) : int(value[3] / self.factor),
                    ] = result_patch[0]

        return self.zero_tensor
