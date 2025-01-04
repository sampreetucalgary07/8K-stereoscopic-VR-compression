import dask.delayed
import os
from tqdm import tqdm
import numpy as np
import dask
import json
import shutil
from dask.diagnostics import ProgressBar  # type: ignore

# User defined libraries
from src.eval_csv import parse_patch_line, read_json
from src.utils.file_utils import make_folder
from src.utils.patch_utils import get_patchSize_list, get_sub_image
from src.utils.tensor_utils import tensor_to_image, image_to_tensor
from src.frames_to_video import patches_to_video_ffmpeg
from src.ffmpeg_video_func import video_to_frames


def get_patchnum_list(percentile_value, metric_matrix, annot_matrix):
    """
    Function to get the list of patch numbers based on the percentile value."""
    patch_num_list = []
    percentile_value = np.percentile(metric_matrix.flatten(), percentile_value)
    print(f"Percentile value: {percentile_value}")
    for i in range(metric_matrix.shape[0]):
        for j in range(metric_matrix.shape[1]):
            if metric_matrix[i][j] > percentile_value:
                patch_num_list.append(
                    int(annot_matrix[i][j].split("\n")[0].split(" ")[1]))

    return patch_num_list


@dask.delayed  # type: ignore
def daskjobs_save_replace_patches(patch_num,frame_no, img_dim, input_true_frame_path,
                                 downsample_factor, output_raw_path):
    patch_size = get_patchSize_list(img_dim, downsample_factor)[f"Patch_{patch_num}"]
    # print(f"Patch size: {patch_size}")
    sub_image = get_sub_image(input_true_frame_path,
                              patch_size, is_tensor=False)
    tensor_to_image(sub_image,
                    save_path=os.path.join(
                        output_raw_path, f"{frame_no}.{os.path.basename(input_true_frame_path).split('.')[1]}"),
                    permute_values=[0, 1, 2])
    return patch_size

def save_replace_patches(input_true_frame_path,
                        patch_num_list, output_raw_path,
                        img_dim, downsample_factor):
    """
    Functionality to save the faulty patches.
    """
    # Get the patch size
    delayed_jobs = []
    for enum,patch_num in enumerate(tqdm(patch_num_list, total=len(patch_num_list), desc="Processing patches")):
        delayed_jobs.append(
            daskjobs_save_replace_patches(patch_num,f"p_{enum:06d}",
                                         img_dim, input_true_frame_path,
                                         downsample_factor,
                                         output_raw_path, )
        )
    with ProgressBar():
        results = dask.compute(*delayed_jobs)  # type: ignore

    #create a .json file at the output_raw_path and save the patch_sizes
    json_path = os.path.join(output_raw_path, "patch_sizes.json")
    with open(json_path, "w") as f:
        json.dump(results, f)
    return json_path
# replacing the default patches with the original patches


def patch_replacement_raw_frame(input_pred_frame_path, video_path, patch_size_path, output_frames_path):



    pred_frame_tensor = image_to_tensor(input_pred_frame_path, normalize=False)
    pred_frame_tensor = pred_frame_tensor.squeeze(0)
    #load the frames from the video and patches from the patch_size_path
    video_frames_path = video_to_frames(
        mp4_file_path = video_path,
        output_folder_name="temp_frames",
        frame_rate=30,
        out_folder_path=output_frames_path,
        return_dest_path=True,
    )   
    patch_sizes = read_json(patch_size_path)

    assert len(os.listdir(video_frames_path)) == len((patch_sizes)), "Number of frames and patches do not match!"

    for patch, patch_size in zip(sorted(os.listdir(video_frames_path)), patch_sizes):
        sub_image = image_to_tensor(os.path.join(video_frames_path, patch), normalize=False) #type: ignore
        pred_frame_tensor[:, patch_size[0]:patch_size[1],
                      patch_size[2]:patch_size[3]] = sub_image

    pred_frame_tensor.unsqueeze_(0)
    tensor_to_image(pred_frame_tensor,
                    save_path=os.path.join(
                        output_frames_path, f"repl_{os.path.basename(input_pred_frame_path)}"),
                    permute_values=[0, 1, 2, 3])

    #remove the temporary frames folder
    print(f"Removing the temporary frames folder: {os.path.join(output_frames_path, 'temp_frames')}")
    shutil.rmtree(os.path.join(output_frames_path, "temp_frames"))

def PatchReplace(input_true_frames_path, input_pred_frames_path, percentile_value,
                 eval_json_path, output_raw_path,
                 output_frames_path,output_video_path, num_frames,
                 metric="lpips",
                 img_dim=(3840, 7680),
                 downsample_factor=40):
    """
    Function to get the original patches. 
    """
    print(f"Input frames path : {input_true_frames_path}")
    data_list = read_json(eval_json_path)
    count = 0
    print("\nSelected no. of frames", len(
        data_list[num_frames[0]:num_frames[1]]))
    for line_list in data_list[num_frames[0]:]:
        print(f"\n-------------------------------------------------")
        print(f"\nProcessing frame : {line_list['true_frame_path']}")

        metric_matrix, annot_matrix, _ = parse_patch_line(
            line_list, downsample_factor, metric
        )
        patch_num_list = get_patchnum_list(
            percentile_value, metric_matrix, annot_matrix)
        input_true_frame_path = os.path.join(
            input_true_frames_path, line_list["true_frame_path"])
        input_pred_frame_path = os.path.join(
            input_pred_frames_path, line_list["pred_frame_path"])
        
        #create a folder to save the patch_list with the original patches
        patch_folder = make_folder(folder_name=line_list["true_frame_path"].split(".")[0], folder_path=output_raw_path,
                    remove_if_exists=True)
        
        print(f"Patch folder: {patch_folder}")
        # Get the original patches
        patch_size_path = save_replace_patches(input_true_frame_path, patch_num_list,
                          patch_folder, img_dim, downsample_factor)
        
        print(f"Patch size path: {patch_size_path}")

        print(f"Output video path: {output_video_path}")
        
        #save the frames in a .mp4 file
        patches_to_video_ffmpeg(
            frames_path = patch_folder,
            output_video_name = "repl_"+line_list["true_frame_path"].split(".")[0],
            video_ext = ".mp4",
            fps = 30,
            img_format=".png",
            num_frames = [None, None],
            out_folder_path=output_video_path,
            overwrite_previous_video=True,
        )

        video_path = os.path.join(output_video_path, "repl_"+line_list["true_frame_path"].split(".")[0]+".mp4")

        # Replace the patches in the frames
        patch_replacement_raw_frame(input_pred_frame_path, video_path, patch_size_path, output_frames_path)


        count += 1
        if count == num_frames[1]:
            break