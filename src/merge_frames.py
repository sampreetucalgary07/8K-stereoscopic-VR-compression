import dask
import shutil
import glob
import dask
import subprocess
from dask.diagnostics import ProgressBar  # type: ignore
from tqdm import tqdm
import os
from PIL import Image
import time
import sys


sys.path.append(os.path.dirname(__file__))
# importing user defined functions
from utils.file_utils import make_folder, filenames
from utils.frame_utils import get_frame_dim


@dask.delayed  # type: ignore
def process_one_frame(
    image_pair, top_down, number, destinationFolder, img_format, frame_info
):
    ext_1_image = Image.open(image_pair[0])
    ext_2_image = Image.open(image_pair[1])
    if top_down:
        new_frame = Image.new(
            frame_info["channels"], (frame_info["width"], frame_info["height"] * 2)
        )
        new_frame.paste(ext_1_image, (0, 0))
        new_frame.paste(ext_2_image, (0, frame_info["height"]))
    else:
        new_frame = Image.new(
            frame_info["channels"], (frame_info["width"] * 2, frame_info["height"])
        )
        new_frame.paste(ext_1_image, (0, 0))
        new_frame.paste(ext_2_image, (frame_info["width"], 0))

    formatted_number = "{:07}".format(number)
    new_frame.save(destinationFolder + "/recon_" + str(formatted_number) + img_format)


def merge_LR_frames(
    frames_path,
    out_folder_name,
    diff_second_frame: bool,
    second_frames_path=None,
    img_format=".jpg",
    out_folder_path=None,
    is_top_down=False,
    num_frames: list = [None, None],
    return_dest_path=True,
):
    """Method used to merge L/T and R/B -eye frames from one frame and save it"""
    if frames_path == None or out_folder_name == None:
        raise ValueError("frames_path or out_folder_name cannot be None")

    _, _, base_folder = filenames(frames_path)

    if out_folder_path != None:
        base_folder = out_folder_path

    destinationFolder = make_folder(
        folder_name=out_folder_name, folder_path=base_folder, remove_if_exists=True
    )

    if is_top_down:
        ext_1 = "T"
        ext_2 = "B"
    else:
        ext_1 = "L"
        ext_2 = "R"

    frames = sorted(glob.glob(frames_path + "/*" + img_format))[
        num_frames[0] : num_frames[1]
    ]
    frame_info = get_frame_dim(frames[0], is_tensor=False)

    if diff_second_frame:
        frames_2 = sorted(glob.glob(os.path.join(second_frames_path, "*",img_format)))[ # type: ignore
            num_frames[0] : num_frames[1]
        ]
        frame_info_2 = get_frame_dim(frames_2[0], is_tensor=False)
        if frame_info != frame_info_2:
            raise ValueError("The frames from the two paths have different dimensions")

    print("\nFrame Info : ", frame_info)
    number = 1
    print(
        f"\nMerging seperate {ext_1} eye and {ext_2} eye frames from the given frames...\n"
    )
    delayed_tasks = []
    for frame_path in tqdm(
        frames,
        desc="Appending dask tasks... ",
        ascii=False,
        ncols=100,
        total=len(frames),
    ):
        # print(frame_path)

        if frame_path.endswith(ext_1 + img_format):
            # print(frame_path)
            ext_1_frame_path = frame_path
            ext_2_frame_path = frame_path.replace(
                ext_1 + img_format, ext_2 + img_format
            )
            if os.path.exists(ext_2_frame_path):
                frame_pair = [ext_1_frame_path, ext_2_frame_path]
                delayed_tasks.append(
                    process_one_frame(
                        frame_pair,
                        is_top_down,
                        number,
                        destinationFolder,
                        img_format,
                        frame_info,
                    )
                )

                number += 1

    print("\n Parallel Processing ... ")
    with ProgressBar():
        dask.compute(*delayed_tasks)  # type: ignore
    print(
        f"\n{ext_1} eye and {ext_2} eye merged frames saved in the folder ",
        destinationFolder,
    )
    if return_dest_path:
        return destinationFolder
