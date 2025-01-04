""" This script generates latent tensors from a dataset using a model"""

### importing standard libraries


from tqdm import tqdm
import os
import numpy as np
import json

# from dask.diagnostics import ProgressBar
from torch.utils.data import DataLoader  # type: ignore
from torchvision import transforms

### importing user defined libraries
from src.utils.patch_utils import get_patchSize_list, get_sub_image
from src.metrics import calculate_ssim, calculate_psnr, calculate_lpips
from src.utils.tensor_utils import path_to_imagelist
from src.utils.file_utils import save_dict_to_file
from src.dataLoader import createTorchDataset
from model.lpips import LPIPS

lpips_model = LPIPS().eval().to("cuda")


# @dask.delayed  # type: ignore
def eval_single_patch(
    true_frame_path,
    pred_frame_path,
    patch_wise,
    patch_list_value,
    metrics_list: list,
    is_tensor: bool = False,
    device: str = "cuda",
):
    store_dict = {}
    if patch_wise:
        sub_img_true = get_sub_image(
            true_frame_path, patch_list_value, False, is_tensor, device
        ).unsqueeze(0)

        sub_img_pred = get_sub_image(
            pred_frame_path, patch_list_value, False, is_tensor, device
        ).unsqueeze(0)
    else:
        sub_img_true = true_frame_path
        sub_img_pred = pred_frame_path
    for metric in metrics_list:
        if metric == "ssim":
            ssim = calculate_ssim(
                sub_img_true, sub_img_pred, is_tensor=True, device=device
            ).get_result()
            store_dict[metric] = round(ssim, 4)
        if metric == "psnr":
            psnr = calculate_psnr(
                sub_img_true, sub_img_pred, is_tensor=True, device=device
            ).get_result()
            store_dict[metric] = round(psnr, 4)
        if metric == "lpips":
            lpips = calculate_lpips(
                lpips_model, sub_img_true, sub_img_pred, is_tensor=True, device=device
            ).get_result()
            store_dict[metric] = round(lpips, 4)
    return store_dict


def eval_frame(
    true_frame_path,
    pred_frame_path,
    patch_wise,
    img_dim,
    downsample_factor,
    metrics_list=["ssim", "psnr"],
    device="cuda",
):
    if patch_wise:
        patch_list = get_patchSize_list(img_dim, downsample_factor)
        delayed_tasks = {}
        for key, value in tqdm(
            patch_list.items(),
            total=len(patch_list),
            desc="Evaluating patches of frames :",
        ):

            res_dict = eval_single_patch(
                true_frame_path,
                pred_frame_path,
                patch_wise,
                value,
                metrics_list,
                is_tensor=True,
                device=device,
            )
            delayed_tasks[key] = res_dict

    else:
        delayed_tasks = eval_single_patch(
            true_frame_path,
            pred_frame_path,
            patch_wise,
            None,
            metrics_list,
            is_tensor=True,
            device=device,
        )

    # print("\n---------Computing the dask graph... ----------\n")
    # print(delayed_tasks.items())
    results = [{key: value} for key, value in delayed_tasks.items()]
    return results


def eval_frame_all_dataloader(
    true_frames_folder_path,
    pred_frames_folder_path,
    res_path,
    img_dim,
    downsample_factor,
    metrics_list,
    num_frames,
    device="cuda",
    patch_wise=False,
):
    json_data = []
    if num_frames == None:
        num_frames = len(path_to_imagelist(pred_frames_folder_path))

    # true_frames_list = path_to_imagelist(true_frames_folder_path)[:num_frames]
    # pred_frames_list = path_to_imagelist(pred_frames_folder_path)[:num_frames]
    sample_true_dataset = createTorchDataset(
        true_frames_folder_path,
        start_num_image=0,
        end_num_image=num_frames,
        transforms=[transforms.ToTensor()],
        channel_last=False,
    )
    true_dataset = DataLoader(
        sample_true_dataset,
        batch_size=1,
        shuffle=False,
    )

    sample_pred_dataset = createTorchDataset(
        pred_frames_folder_path,
        start_num_image=0,
        end_num_image=num_frames,
        transforms=[transforms.ToTensor()],
        channel_last=False,
    )
    pred_dataset = DataLoader(
        sample_pred_dataset,
        batch_size=1,
        shuffle=False,
    )

    assert len(sample_true_dataset) == len(
        sample_pred_dataset
    ), "----- Number of frames in the true and predicted folders should be the same! ----"

    for idx, (
        (true_frame_tensor, true_frame_path),
        (pred_frame_tensor, pred_frame_path),
    ) in enumerate(
        zip(
            true_dataset,
            pred_dataset,
        )
    ):
        true_frame_path = true_frame_path[0]
        pred_frame_path = pred_frame_path[0]
        print(
            f"\n [{idx +1}/{num_frames}].Evaluating frame : {os.path.basename(true_frame_path)} and {os.path.basename(pred_frame_path)}\n"
        )
        assert (
            true_frame_path[-10:] == pred_frame_path[-10:]
        ), "True and pred frame numbers do not match"

        res_log = eval_frame(
            true_frame_tensor,
            pred_frame_tensor,
            patch_wise,
            img_dim,
            downsample_factor,
            metrics_list,
            device,
        )
        # save results
        # save_dict_to_file(res_log, os.path.basename(true_frame_path), res_path)
        data = {
            "true_frame_path": os.path.basename(true_frame_path),
            "pred_frame_path": os.path.basename(pred_frame_path),
            "patch_metric": res_log,
        }

        json_data.append(data)

    with open(res_path, "w") as f:
        json.dump(json_data, f, indent=4)

        # with open(res_path, "a") as f:
        #     f.write(
        #         f"{os.path.basename(true_frame_path)},{os.path.basename(pred_frame_path)},{res_log}\n"
        #     )
