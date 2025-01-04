# This .py file takes the folder containing L and R-eye frames and split it into train and test.

# importing libraries
import dask
import shutil
import glob
import dask
from dask.diagnostics import ProgressBar  # type: ignore
from tqdm import tqdm
import math
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image

import sys


sys.path.append(os.path.dirname(__file__))
# importing user defined functions
from utils.file_utils import make_folder, filenames

# from evaluation import *


@dask.delayed  # type: ignore
def process_one_image(image_path, top_down, number, destinationFolder, img_format):
    image = Image.open(image_path)
    width, height = image.size
    if top_down:
        left_half = (0, 0, width, height // 2)
        right_half = (0, height // 2, width, height)
        ext_1 = "T"
        ext_2 = "B"
    else:
        left_half = (0, 0, width // 2, height)
        right_half = (width // 2, 0, width, height)
        ext_1 = "L"
        ext_2 = "R"
    left_image = image.crop(left_half)
    right_image = image.crop(right_half)
    # formatted_number = "{:07}".format(number)

    left_image.save(
        os.path.join(destinationFolder, os.path.basename(image_path).split(".")[0])
        + ext_1
        + img_format
    )
    right_image.save(
        os.path.join(destinationFolder, os.path.basename(image_path).split(".")[0])
        + ext_2
        + img_format
    )


def split_LR_frames(
    frames_path,
    out_folder_name,
    img_format=".jpg",
    out_folder_path=None,
    is_top_down=False,
    num_frames: list = [0, -1],
):
    """Method used to seperate L and R -eye frames from one frame and save it"""
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

    number = 1
    print(
        f"\nExtracting seperate {ext_1} eye and {ext_2} frames from the given frames...\n"
    )
    delayed_tasks = []
    for image_path in tqdm(
        frames, desc="Processing ... ", ascii=False, ncols=100, total=len(frames)
    ):
        delayed_tasks.append(
            process_one_image(
                image_path, is_top_down, number, destinationFolder, img_format
            )
        )

        number += 1
    print("\n Parallel Processing ... ")
    with ProgressBar():
        dask.compute(*delayed_tasks)  # type: ignore
    print(
        f"\n{ext_1} eye and {ext_2} separate frames saved in the folder ",
        destinationFolder,
    )


def LR_train_test_split(
    LR_frames_path,
    perc_of_train=0.75,
    num_images=-0.5,  # -0.5 means all images TO-DO logic for num frames
    img_format=".png",
    out_folder_path=None,
    seperate_L_R=False,
):
    """Class to separate the L and R frames and save into a seperate folder"""
    if LR_frames_path == None:
        raise ValueError("LR_frames_path cannot be None")

    _, _, base_folder = filenames(LR_frames_path)

    if out_folder_path != None:
        base_folder = out_folder_path

    frames = sorted(glob.glob(LR_frames_path + "/*" + img_format))[
        : int(num_images * 2)
    ]
    # print(frames)
    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError("No frames found in the given path")

    if int(math.ceil(num_frames * perc_of_train)) % 2 == 0:

        train_range = frames[: int(math.ceil(num_frames * perc_of_train))]
        test_range = frames[int(math.ceil(num_frames * perc_of_train)) :]
    else:

        train_range = frames[: int(math.ceil(num_frames * perc_of_train)) + 1]
        test_range = frames[int(math.ceil(num_frames * perc_of_train)) + 1 :]

    print("\nTotal frames : ", num_frames)
    print(
        f"Percentage of frames used to train : {perc_of_train*100} %",
    )
    print("Total Train frames : ", len(train_range))
    print("Total Test frames : ", len(test_range))
    print(
        f"Train frames are between {os.path.basename(train_range[0])} and {os.path.basename(train_range[-1])}"
    )
    print(
        f"Test frames are between {os.path.basename(test_range[0])} and {os.path.basename(test_range[-1])}\n"
    )
    if seperate_L_R:

        train_L = make_folder(
            folder_name="trainL", folder_path=base_folder, remove_if_exists=True
        )
        train_R = make_folder(
            folder_name="trainR", folder_path=base_folder, remove_if_exists=True
        )
        test_L = make_folder(
            folder_name="testL", folder_path=base_folder, remove_if_exists=True
        )
        test_R = make_folder(
            folder_name="testR", folder_path=base_folder, remove_if_exists=True
        )

        print("\nRespective folders made to save separate frames...")

        for frame in tqdm(
            train_range, total=len(train_range), desc="Copying Train frames"
        ):
            # print(frame)
            if frame[-5:] == "L" + img_format:
                frame_path = os.path.join(LR_frames_path, frame)
                shutil.copy(frame_path, train_L)
            else:
                frame_path = os.path.join(LR_frames_path, frame)
                shutil.copy(frame_path, train_R)

        print(f"\nTrain frames copied!")
        print(
            f"TrainL frames are between {os.path.basename(sorted(os.listdir(train_L))[0])} and {os.path.basename(sorted(os.listdir(train_L))[-1])}"
        )
        print(
            f"TrainR frames are between {os.path.basename(sorted(os.listdir(train_R))[0])} and {os.path.basename(sorted(os.listdir(train_R))[-1])}\n"
        )
        for frame in tqdm(
            test_range, total=len(test_range), desc="Copying Test frames"
        ):
            # print(frame)
            if frame[-5:] == "L" + img_format:
                # frame_path = os.path.join(LR_frames_path, frame)
                shutil.copy(frame, test_L)
            else:
                # frame_path = os.path.join(LR_frames_path, frame)
                shutil.copy(frame, test_R)

        print(f"\nTest frames copied!")
        print(
            f"TestL frames are between {os.path.basename(sorted(os.listdir(test_L))[0])} and {os.path.basename(sorted(os.listdir(test_L))[-1])}"
        )
        print(
            f"TestR frames are between {os.path.basename(sorted(os.listdir(test_R))[0])} and {os.path.basename(sorted(os.listdir(test_R))[-1])}"
        )

        return train_L, train_R, test_L, test_R

    else:
        train_frames = make_folder(
            folder_name="train_frames", folder_path=base_folder, remove_if_exists=True
        )

        test_frames = make_folder(
            folder_name="test_frames", folder_path=base_folder, remove_if_exists=True
        )

        for frame in tqdm(
            train_range, total=len(train_range), desc="Copying Train frames..."
        ):

            frame_path = os.path.join(LR_frames_path, frame)
            shutil.copy(frame_path, train_frames)

        for frame in tqdm(
            test_range, total=len(test_range), desc="Copying Test frames..."
        ):

            frame_path = os.path.join(LR_frames_path, frame)
            shutil.copy(frame_path, test_frames)

        return train_frames, test_frames
