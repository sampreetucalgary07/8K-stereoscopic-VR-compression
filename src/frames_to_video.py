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

ffmpeg = "/usr/bin/ffmpeg"
ffprobe = "/usr/bin/ffprobe"


def frames_to_video_ffmpeg(
    frames_path,
    output_video_name,
    video_ext,
    in_fps,
    out_fps,
    video_codec,
    pixel_format,
    img_format=".jpg",
    crf_value=23,
    num_frames: list = [None, None],
    audio_codec=None,
    audio_path=None,
    out_folder_path=None,
    overwrite_previous_video=False,
):
    """
    Function to generate video from frames
    """
    if frames_path == None or output_video_name == None:
        raise ValueError("frames_path or output_video_name cannot be None")

    _, _, base_folder = filenames(frames_path)

    if out_folder_path != None:
        base_folder = out_folder_path

    output_video_name = output_video_name + video_ext
    output_video_name = os.path.join(base_folder, output_video_name)

    if os.path.exists(output_video_name):
        if overwrite_previous_video:
            os.remove(output_video_name)
            print("Previous video deleted successfully at the same location!")
        else:
            raise ValueError(
                "Video already exists at the given path. Set overwrite_previous_video=True to delete the previous video and create a new one."
            )

    else:
        print("No previous video found with the same name at the location!!")
    temp_folder = make_folder(
        folder_name="TempFolder", folder_path=base_folder, remove_if_exists=True
    )

    # copy the frames from the frames path to the temp folder
    for frame in sorted(os.listdir(frames_path))[num_frames[0] : num_frames[1]]:
        if frame.endswith(img_format):
            shutil.copy2(
                os.path.join(frames_path, frame),
                temp_folder,
            )
    print("Frames copied to a temporary folder successfully!")
    print(f"Frames copied at: ", temp_folder)

    # rename the frames in the temp folder from out_0000001.jpg to 0000001.jpg
    for i, filename in enumerate(sorted(os.listdir(temp_folder))):
        os.rename(temp_folder + "/" + filename, temp_folder + "/" + str(i) + img_format)

    print("Frames renamed successfully for compatible with ffmpeg in a sorted manner!")

    frames_path = os.path.join(temp_folder, "%d" + img_format)

    if audio_codec == None or audio_path == None:
        print("Audio path or Audio codec not provided!")
        print("Generating video without audio...")
        start_time = time.time()
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-framerate",
            str(in_fps),
            "-i",
            frames_path,
            "-c:v",
            video_codec,
            "-crf",
            str(crf_value),
            "-r",
            str(out_fps),
            "-pix_fmt",
            pixel_format,  # Pixel format
            output_video_name,  # Output path
        ]
        total_time = time.time() - start_time

    else:
        start_time = time.time()
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-framerate",
            str(in_fps),
            "-i",
            frames_path,  # Input frames
            "-i",
            audio_path,  # Input audio
            "-c:v",
            video_codec,
            "-crf",
            str(crf_value),  # Video codec
            "-r",
            str(out_fps),
            "-c:a",
            audio_codec,  # Audio codec
            "-strict",
            "experimental",
            "-pix_fmt",
            pixel_format,  # Pixel format
            output_video_name,  # Output path
        ]
        total_time = time.time() - start_time
    print("\n -------- Making video from frames... ----------\n")
    subprocess.run(command)
    print("\n---------- Video created successfully! ----------")
    print(f"\nVideo saved at : {output_video_name}")

    print("\nTotal time taken for encoding: ", total_time, " seconds")
    print(
        "\nFile size: ",
        os.path.getsize(output_video_name) / (1024 * 1024),
        "MB",
    )
    print("\nDeleting temporary folder...")
    shutil.rmtree(temp_folder)


def latents_to_video_ffmpeg(
    frames_path,
    output_video_name,
    video_ext,
    in_fps,
    img_format=".png",
    num_frames: list = [None, None],
    out_folder_path=None,
    overwrite_previous_video=False,
    preset = "veryslow",
    qp = 0,
):
    """
    Function to generate video from frames
    """
    if frames_path == None or output_video_name == None:
        raise ValueError("frames_path or output_video_name cannot be None")

    _, _, base_folder = filenames(frames_path)

    if out_folder_path != None:
        base_folder = out_folder_path

    output_video_name = output_video_name + video_ext
    output_video_name = os.path.join(base_folder, output_video_name)

    if os.path.exists(output_video_name):
        if overwrite_previous_video:
            os.remove(output_video_name)
            print("Previous video deleted successfully at the same location!")
        else:
            raise ValueError(
                "Video already exists at the given path. Set overwrite_previous_video=True to delete the previous video and create a new one."
            )

    else:
        print("No previous video found with the same name at the location!!")
    temp_folder = make_folder(
        folder_name="TempFolder", folder_path=base_folder, remove_if_exists=True
    )

    # copy the frames from the frames path to the temp folder
    for frame in sorted(os.listdir(frames_path))[num_frames[0] : num_frames[1]]:
        if frame.endswith(img_format):
            shutil.copy2(
                os.path.join(frames_path, frame),
                temp_folder,
            )
    print("Frames copied to a temporary folder successfully!")
    print(f"Frames copied at: ", temp_folder)

    # rename the frames in the temp folder from out_0000001.jpg to 0000001.jpg
    for i, filename in enumerate(sorted(os.listdir(temp_folder))):
        new_name = f"s{i:06d}{img_format}"
        os.rename(
            os.path.join(temp_folder, filename), os.path.join(temp_folder, new_name)
        )

    print("Frames renamed successfully for compatible with ffmpeg in a sorted manner!")

    frames_path = os.path.join(temp_folder, "s%06d" + img_format)

    start_time = time.time()
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-framerate",
        str(in_fps),
        "-i",
        frames_path,  # Input frames
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-qp",
        str(qp),
        "-pix_fmt",
        "yuv444p",
        output_video_name,  # Output path
    ]
    total_time = time.time() - start_time
    print("\n -------- Making video from frames... ----------\n")
    subprocess.run(command)
    print("\n---------- Video created successfully! ----------")
    print(f"\nVideo saved at : {output_video_name}")

    print("\nTotal time taken for encoding: ", total_time, " seconds")
    print(
        "\nFile size: ",
        os.path.getsize(output_video_name) / (1024 * 1024),
        "MB",
    )
    print("\nDeleting temporary folder...")
    shutil.rmtree(temp_folder)


def patches_to_video_ffmpeg(
    frames_path,
    output_video_name,
    video_ext,
    fps,
    preset = "medium",
    img_format=".jpg",
    num_frames: list = [None, None],
    out_folder_path=None,
    overwrite_previous_video=False,
):
    """
    Function to generate video from frames
    """
    if frames_path == None or output_video_name == None:
        raise ValueError("frames_path or output_video_name cannot be None")

    _, _, base_folder = filenames(frames_path)

    if out_folder_path != None:
        base_folder = out_folder_path

    output_video_name = output_video_name + video_ext
    output_video_name = os.path.join(base_folder, output_video_name)

    if os.path.exists(output_video_name):
        if overwrite_previous_video:
            os.remove(output_video_name)
            print("Previous video deleted successfully at the same location!")
        else:
            raise ValueError(
                "Video already exists at the given path. Set overwrite_previous_video=True to delete the previous video and create a new one."
            )

    else:
        print("No previous video found with the same name at the location!!")
    temp_folder = make_folder(
        folder_name="TempFolder", folder_path=base_folder, remove_if_exists=True
    )

    # copy the frames from the frames path to the temp folder
    for frame in sorted(os.listdir(frames_path))[num_frames[0] : num_frames[1]]:
        if frame.endswith(img_format):
            shutil.copy2(
                os.path.join(frames_path, frame),
                temp_folder,
            )
    print("Frames copied to a temporary folder successfully!")
    print(f"Frames copied at: ", temp_folder)

    # rename the frames in the temp folder from out_0000001.jpg to 0000001.jpg
    for i, filename in enumerate(sorted(os.listdir(temp_folder))):
        new_name = f"s{i:06d}{img_format}"
        os.rename(
            os.path.join(temp_folder, filename), os.path.join(temp_folder, new_name)
        )

    print("Frames renamed successfully for compatible with ffmpeg in a sorted manner!")

    frames_path = os.path.join(temp_folder, "s%06d" + img_format)

    print("Audio path or Audio codec not provided!")
    print("Generating video without audio...")
    start_time = time.time()
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        frames_path,  # Input frames
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        output_video_name  # Output path
    ]
        

    total_time = time.time() - start_time
    print("\n -------- Making video from frames... ----------\n")
    subprocess.run(command)
    print("\n---------- Video created successfully! ----------")
    print(f"\nVideo saved at : {output_video_name}")

    print("\nTotal time taken for encoding: ", total_time, " seconds")
    print(
        "\nFile size: ",
        os.path.getsize(output_video_name) / (1024 * 1024),
        "MB",
    )
    print("\nDeleting temporary folder...")
    shutil.rmtree(temp_folder)