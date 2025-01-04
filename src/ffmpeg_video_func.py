import subprocess
import os
import sys

sys.path.append(os.path.dirname(__file__))
from .utils.file_utils import filenames, make_folder
import time
import json


ffmpeg = "/usr/bin/ffmpeg"
ffprobe = "/usr/bin/ffprobe"


def video_info(mp4_file_path, return_values=False):
    """
    Function to get Video file info
    """
    name, ext, _ = filenames(mp4_file_path)
    command_video = [
        ffprobe,
        "-loglevel",
        "error",
        "-v",
        "error",  # Hide unnecessary warnings
        "-select_streams",
        "v:0",  # Select the first video stream
        "-show_entries",
        "stream=r_frame_rate,codec_name,pix_fmt,duration",  # Show the real frame rate
        "-of",
        "json",  # Output as JSON
        mp4_file_path,
    ]
    print("Decoding Video information ...")
    result = subprocess.run(
        command_video, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output_json = json.loads(result.stdout.decode())
    video_codec = output_json["streams"][0]["codec_name"]
    pixel_format = output_json["streams"][0]["pix_fmt"]
    r_frame_rate = output_json["streams"][0]["r_frame_rate"]
    dur = output_json["streams"][0]["duration"]
    numerator, denominator = map(int, r_frame_rate.split("/"))
    fps = round(numerator / denominator)

    print(f"Below are the specs for the video {name + ext}")
    print(f"Frames per seconds : ", fps)
    print(f"Video Codec : ", video_codec)
    print(f"Pixel Format : ", pixel_format)
    print(f"Duration : ", dur, "seconds")

    if return_values:
        return fps, video_codec, pixel_format, dur


def audio_info(mp4_file_path):
    """
    Function to extract audio information
    """
    command_audio = [
        ffprobe,
        "-loglevel",
        "error",
        "-v",
        "error",  # Hide unnecessary warnings
        "-select_streams",
        "a:0",  # Select the first audio stream
        "-show_entries",
        "stream=codec_name",  # Show the codec name
        "-of",
        "json",  # Output as JSON
        mp4_file_path,
    ]
    print("\nDecoding audio information ...")
    try:
        result = subprocess.run(
            command_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output_json = json.loads(result.stdout.decode())
        audio_codec = output_json["streams"][0]["codec_name"]
        print(f"\nAudio Codec : ", audio_codec)
    except:
        print("Audio not present!")
        audio_codec = None

    return audio_codec


def trim_video(
    mp4_file_path, start=0, end=10, out_mp4_path=None, return_dest_path=False
):
    """
    Function to trim the video from {start} seconds to {end} seconds
    """

    name, _, base_folder = filenames(mp4_file_path)

    out_mp4_name = name + f"_trimmed_{start}_{end}.mp4"

    if out_mp4_path == None:
        out_mp4_path = os.path.join(base_folder, out_mp4_name)
    else:
        out_mp4_path = os.path.join(out_mp4_path, out_mp4_name)

    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        mp4_file_path,  # Input video file path
        "-ss",
        str(start),  # Best quality audio
        "-to",
        str(end),  # Extract all audio streams
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        out_mp4_path,  # trimmed output file to be saved
        # Output audio file path
    ]
    print(f"\nTrimmed video saved at {out_mp4_path}")
    subprocess.run(command)
    if return_dest_path:
        return out_mp4_path


def video_to_frames(
    mp4_file_path,
    output_folder_name=None,
    frame_rate=30,
    out_folder_path=None,
    return_dest_path=False,
):
    """
    Any video to frames
    """
    name, _, base_folder = filenames(mp4_file_path)

    if out_folder_path != None:
        base_folder = out_folder_path

    # out_folder_frames = os.path.join(base_folder, name + "_frames")

    if output_folder_name == None:
        output_folder_name = name + "_frames"

    destinationFolder = make_folder(
        folder_name=output_folder_name, folder_path=base_folder
    )

    print("\nExtracting frames from Video : ", name)
    start_time = time.time()
    os.system(
        'ffmpeg -loglevel error -i {} -r {} -vf "scale=sws_flags=lanczos:in_color_matrix=bt709,format=yuv420p" {}/out_%07d.png'.format(
            mp4_file_path, (frame_rate * 1000) / 1001, destinationFolder
        )
    )
    total_time = time.time() - start_time
    print(f"\nVideo frames saved in the folder : ", destinationFolder)
    print(f"\nTime taken to extract frames : {total_time} seconds")

    if return_dest_path:
        return destinationFolder


def video_to_latents(
    mp4_file_path,
    output_folder_name=None,
    frame_rate=30,
    out_folder_path=None,
    return_dest_path=False,
):
    """
    Any video to frames
    """
    name, _, base_folder = filenames(mp4_file_path)

    if out_folder_path != None:
        base_folder = out_folder_path

    # out_folder_frames = os.path.join(base_folder, name + "_frames")

    if output_folder_name == None:
        output_folder_name = name + "_frames"

    destinationFolder = make_folder(
        folder_name=output_folder_name, folder_path=base_folder
    )

    print("\nExtracting frames from Video : ", name)
    start_time = time.time()
    os.system(
        'ffmpeg -loglevel error -i "{}" -r "{}" -pix_fmt yuv444p {}/enc_out_%07d.png'.format(
            mp4_file_path, (frame_rate * 1000) / 1001, destinationFolder
        )
    )
    total_time = time.time() - start_time
    print(f"\nVideo frames saved in the folder : ", destinationFolder)
    print(f"\nTime taken to extract frames : {total_time} seconds")

    if return_dest_path:
        return destinationFolder


def audio_extract(mp4_file_path, out_folder_path=None):
    """
    Audio extraction from video file
    """

    name, _, base_folder = filenames(mp4_file_path)

    if out_folder_path != None:
        base_folder = out_folder_path

    audio_name = name + "_audio" + ".mp3"
    audio_path = os.path.join(base_folder, audio_name)

    try:
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-i",
            mp4_file_path,  # Input video file path
            "-q:a",
            "0",  # Best quality audio
            "-map",
            "a",  # Extract all audio streams
            audio_path,  # Output audio file path
        ]
        print("\nExtracting audio...\n")
        subprocess.run(command)
        print(f"Audio saved as ", audio_path)

    except:
        print("Audio not present!")

    return audio_path


# def leftOnly_frames(output_folder_orig, output_folder_left):
#     """
#     L-R frames to L-L frames conversion
#     """
#     make_folder(output_folder_left)
#     images = sorted(glob.glob("./" + output_folder_orig + "/*.jpg"))
#     number = 1
#     print("\nExtracting L-eye image from the frames...\n")
#     for image_path in tqdm(
#         images, desc="processing", ascii=False, ncols=100, total=len(images)
#     ):
#         image = Image.open(image_path)
#         width, height = image.size
#         left_half = (0, 0, width // 2, height)
#         #right_half = (width // 2, 0, width, height)
#         left_image = image.crop(left_half)
#         # right_image = image.crop(right_half)
#         new_image = Image.new("RGB", (width, height))
#         new_image.paste(left_image, (0, 0))
#         new_image.paste(left_image, (width // 2, 0))
#         formatted_number = "{:05}".format(number)
#         new_image.save(
#             "./" + output_folder_left + "/leftOnly_" + str(formatted_number) + ".jpg"
#         )
#         number += 1
#     print(f"\nL-eye frames saved in the folder ", output_folder_left)
