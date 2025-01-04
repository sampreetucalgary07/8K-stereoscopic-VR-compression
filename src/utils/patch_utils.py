from src.utils.tensor_utils import image_to_tensor
import dask
from PIL import Image


def get_patchSize_list(image_max_dim: tuple, downscale_factor: int, log: bool = False):
    # image_dim = ()
    """Given an image max dim and sub_image_size,
    returns a dictionary with the patch sizes values in a list
    For eg. {'Patch_1': [0, 64, 0, 64]}

    Args:
    image_max_dim : tuple : (H, W)
    downscale_factor: int : (downampling factor = x4/x8/x16)
    log : bool : (print the no. of patches)

    """

    patch_size_list = {}
    n = 0
    for i in range(
        0, list(image_max_dim)[0], int(list(image_max_dim)[0] / downscale_factor)
    ):
        for j in range(
            0, list(image_max_dim)[1], int(list(image_max_dim)[1] / downscale_factor)
        ):
            n += 1
            patch_size_list["Patch_" + str(n)] = list(
                (
                    i,
                    i + int(list(image_max_dim)[0] / downscale_factor),
                    j,
                    j + int(list(image_max_dim)[1] / downscale_factor),
                )
            )

    if log:
        print(f"No. of patch_sizes in patch list : {n}")

    return patch_size_list


# type: ignore
def get_sub_image(
    image,
    patch_list: list,
    print_size: bool = False,
    is_tensor: bool = False,
    device="cuda",
):
    """Funct to get the sub image from a dataloader / img by giving a
    patch_size_list (list of 4 values)"""
    if is_tensor:
        img = image
    else:
        img = image_to_tensor(image, False, device) 

    subImage = img[0][:, patch_list[0] : patch_list[1], patch_list[2] : patch_list[3]]

    if print_size:
        print("Size : ", subImage.size())  # type: ignore
    return subImage
