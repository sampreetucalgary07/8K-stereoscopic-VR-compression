""" This script measures various metrics on the frames"""

## importing standard libraries
import torch
from tqdm import tqdm
import numpy as np
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

## importing custom libraries
from src.utils.tensor_utils import image_to_tensor
from src.utils.file_utils import save_dict_to_file
from model.lpips import LPIPS


############# Metrics functions ####################


class imageMetric:
    def __init__(self, image1, image2, is_tensor=False, device="cuda"):
        self.image1 = image1
        self.image2 = image2
        self.is_tensor = is_tensor
        self.device = device

    def forward_image(self):
        if self.is_tensor:  # If the input is tensor
            self.image1_tensor = self.image1
            self.image2_tensor = self.image2
        else:
            self.image1_tensor = image_to_tensor(self.image1)
            self.image2_tensor = image_to_tensor(self.image2)

        self.image1_tensor = self.image1_tensor.to(self.device)
        self.image2_tensor = self.image2_tensor.to(self.device)

    def calculate_metric(self):
        raise NotImplementedError(
            "Subclasses must implement the calculate_metrics method"
        )

    def get_result(self):
        self.result = self.calculate_metric()
        return self.result


class calculate_ssim(imageMetric):
    def calculate_metric(self):
        self.forward_image()
        ssim = StructuralSimilarityIndexMeasure().to(self.device)(
            self.image1_tensor, self.image2_tensor
        )
        ssim_value = ssim.item()
        if np.isnan(ssim.item()):
            ssim_value = 1.0
        return ssim_value


class calculate_psnr(imageMetric):
    def calculate_metric(self):
        self.forward_image()
        psnr = PeakSignalNoiseRatio().to(self.device)(
            self.image1_tensor, self.image2_tensor
        )
        return psnr.item()


class calculate_lpips(imageMetric):
    def __init__(self, lpips_model, image1, image2, is_tensor=False, device="cuda"):
        super().__init__(image1, image2, is_tensor, device)
        self.lpips_model = lpips_model

    def calculate_metric(self):
        self.forward_image()
        lpips_loss = self.lpips_model(self.image2_tensor, self.image1_tensor)
        return lpips_loss.item()


class calculate_pixel_diff_perc(imageMetric):
    def calculate_metric(self):
        self.forward_image()
        # Calculate pixel difference percentage
        diff = torch.abs(torch.sub(self.image1_tensor, self.image2_tensor))
        diff_perc = (
            torch.sum(diff)
            / (
                self.image1_tensor.size(1)
                * self.image1_tensor.size(2)
                * self.image1_tensor.size(3)
            )
            * 100
        )
        return diff_perc.item()


class ImageMetrics_all:
    def __init__(self, image_list_1, image_list_2, is_tensor=False, device="cuda"):
        self.image_list_1 = image_list_1
        self.image_list_2 = image_list_2
        self.is_tensor = is_tensor
        self.device = device

    def calculate_metrics(self):
        raise NotImplementedError(
            "Subclasses must implement the calculate_metrics method"
        )

    def get_results(self):
        self.results = self.calculate_metrics()
        return self.results

    def save_results(self, file_name, file_path):
        save_dict_to_file(self.results, file_name, file_path)


class calc_ssim_all(ImageMetrics_all):
    def calculate_metrics(self):
        ssim_values = []
        for image1, image2 in tqdm(
            zip(self.image_list_1, self.image_list_2), total=len(self.image_list_1)
        ):
            ssim = calculate_ssim(image1, image2, self.is_tensor, self.device)
            ssim_values.append(ssim.get_result())
        return ssim_values


class calc_psnr_all(ImageMetrics_all):
    def calculate_metrics(self):
        psnr_values = []
        for image1, image2 in tqdm(
            zip(self.image_list_1, self.image_list_2), total=len(self.image_list_1)
        ):
            psnr = calculate_psnr(image1, image2, self.is_tensor, self.device)
            psnr_values.append(psnr.get_result())
        return psnr_values


class calc_pixel_diff_perc_all(ImageMetrics_all):
    def calculate_metrics(self):
        diff_perc_values = []
        for image1, image2 in tqdm(
            zip(self.image_list_1, self.image_list_2), total=len(self.image_list_1)
        ):
            diff_perc = calculate_pixel_diff_perc(
                image1, image2, self.is_tensor, self.device
            )
            diff_perc_values.append(diff_perc.get_result())
        return diff_perc_values
