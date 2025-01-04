import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import sys

sys.path.append(os.path.dirname(__file__))
from utils.file_utils import check_image_size


class createTorchDataset(Dataset):
    def __init__(
        self,
        img_dir,
        start_num_image=None,
        end_num_image=None,
        transforms: list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ],
        channel_last=False,
    ):
        self.img_dir = img_dir
        self.transform = transforms
        self.list_images = sorted(
            file
            for file in os.listdir(self.img_dir)
            if (file.endswith(".jpg") or file.endswith(".png"))
        )
        self.start_num_image = start_num_image
        self.end_num_image = end_num_image

        # print("Directory: ", self.img_dir)
        # print("First image in the directory: ", self.list_images[0])
        # print("Last image in the directory: ", self.list_images[-1])
        # print("Total number of images in the directory: ", len(self.list_images))
        self.list_images = sorted(
            self.list_images[self.start_num_image : self.end_num_image]
        )
        self.channel_last = channel_last

        print("\n ----- DATASET INFO ----- \n")

        print("Selected number of images in the directory: ", len(self.list_images))
        print(f"From image {self.list_images[0]} to image {self.list_images[-1]}\n")

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.list_images[idx])
        image = Image.open(img_path)
        im_tensor = torchvision.transforms.Compose(self.transform)(image)
        if self.channel_last:
            try:
                image = torch.permute(image, (1, 2, 0))  # type: ignore
            except TypeError:
                print("Please convert the image to a torch tensor first! ")

        return im_tensor, img_path
