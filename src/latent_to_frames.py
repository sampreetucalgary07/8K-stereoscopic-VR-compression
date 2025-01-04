""" This src file contains the class model_to_frames which is used to convert a model to latents"""

# standard libraries
import os
import torch
from tqdm import tqdm
import torchvision
import shutil
from PIL import Image
import time

# user defined libraries
from src.patches import frame_from_patches
from src.utils.file_utils import filenames, make_folder
from src.utils.tensor_utils import path_to_imagelist


class latent_to_frame(frame_from_patches):
    def __init__(
        self,
        image,
        patch_size_list,
        model,
        upsample=True,
        factor=1,
        is_tensor=True,
        device="cuda",
    ):
        super().__init__(
            image=image,
            patch_size_list=patch_size_list,
            model=model,
            upsample=upsample,
            factor=factor,
            is_tensor=is_tensor,
            device=device,
        )

    def preprocess_patch(self, patch):
        patch = patch.unsqueeze(0)
        return patch

    def get_model_output_patch(self, patch):
        decoded_output = self.model.decode(patch.to(self.device))
        decoded_output = torch.clamp(decoded_output, -1.0, 1.0)
        decoded_output = (decoded_output + 1) / 2
        return decoded_output


def save_raw_frame_all(
    inp_latent_folder,
    model,
    destination_path,
    output_folder_name,
    patch_list,
    num_images:list,
    upsample=True,
    factor=8,
    img_format=".png",
    device="cuda",
):
    # print("path : ", inp_latent_folder)
    foldername, _, base_path = filenames(inp_latent_folder)

    if destination_path == None:
        destination_path = base_path

    if output_folder_name:
        foldername = output_folder_name
    else:
        foldername = foldername + "_latent_tensors"

    dest_folder = make_folder(
        foldername,
        folder_path=destination_path,
        remove_if_exists=True,
    )
    # if num_images == -1:
    #     imageList = path_to_imagelist(inp_latent_folder)[:]
    # else:
    imageList = path_to_imagelist(inp_latent_folder)[num_images[0]:num_images[1]]

    # print(imageList)
    for lat in tqdm(imageList, total=len(imageList), desc="Saving Frames..."):
        decoded_tensor = latent_to_frame(
            torch.load(lat),
            patch_list,
            model,
            upsample=upsample,
            factor=factor,
            is_tensor=True,
            device=device,
        ).get_result()
        # remove .jpg from base_name

        save_path = os.path.join(
            dest_folder,
            "dec_" + os.path.splitext(os.path.basename(lat))[0] + img_format,
        )

        # tensor_to_image(encoded_image, save_path, permute_values=[0, 1, 2, 3])
        torchvision.utils.save_image(
            decoded_tensor,
            save_path,
        )

    print("\nFrames saved to: ", dest_folder)
    return dest_folder


############################################################################################################


class vae_frame_from_patches(frame_from_patches):
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
        super().__init__(
            image=image,
            patch_size_list=patch_size_list,
            model=model,
            upsample=upsample,
            factor=factor,
            is_tensor=is_tensor,
            device=device,
        )

    def preprocess_patch(self, patch):
        # patch = torchvision.transforms.ToTensor()(patch)
        patch = (2 * patch) - 1
        patch = patch.unsqueeze(0)
        return patch

    def get_model_output_patch(self, patch):
        encoded_output, _ = self.model.encode(patch.to(self.device))
        decoded_output = self.model.decode(encoded_output)
        decoded_output = torch.clamp(decoded_output, -1.0, 1.0)
        decoded_output = (decoded_output + 1) / 2
        return decoded_output


class model_to_frames:
    def __init__(
        self,
        input_folder_path,
        output_folder_path,
        out_width,
        out_height,
        model,
        frame_ext=".jpg",
        L_frame_path=None,
        model_parameters_path=None,
        delete_temp=False,
        device="cuda",
    ):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.out_width = out_width
        self.out_height = out_height
        self.model = model
        self.model_parameters_path = model_parameters_path
        self.delete_temp = delete_temp
        self.device = device
        self.process_all_images_time = 0
        # list of frames
        if L_frame_path == None:
            self.L_frame_path = self.input_folder_path
        else:
            self.L_frame_path = L_frame_path
        self.frame_list = sorted(os.listdir(self.input_folder_path))
        self.L_frame_list = sorted(os.listdir(self.L_frame_path))
        self.frame_ext = frame_ext
        # make temp folder
        self.temp_folder_path = make_folder(
            folder_name="temp_pred_frames",
            folder_path=self.output_folder_path,
            remove_if_exists=True,
        )

        # make output folder
        self.LR_folder_path = make_folder(
            folder_name="LR_pred_frames",
            folder_path=self.output_folder_path,
            remove_if_exists=True,
        )

        self.processed_model = self.process_model()

    def process_model(self):
        raise NotImplementedError

    def process_input_for_model(self, in_path):
        raise NotImplementedError

    def model_output(self, processed_input):
        raise NotImplementedError

    def process_model_output(self, raw_model_output):
        raise NotImplementedError

    def process_all_inputs(self):
        with torch.no_grad():
            for idx, inp in enumerate(
                tqdm(
                    self.frame_list,
                    total=len(self.frame_list),
                    desc="Processing Frames",
                )
            ):

                in_frame_path = os.path.join(self.input_folder_path, inp)
                # print(in_frame_path)
                processed_input = self.process_input_for_model(in_frame_path)
                # print(processed_input)
                raw_model_output = self.model_output(processed_input)
                # print(raw_model_output.shape)
                processed_output = self.process_model_output(raw_model_output)
                # print(processed_output.shape)

                out_frame_path = os.path.join(
                    self.temp_folder_path,
                    inp.split(".")[0].replace("L", "R") + self.frame_ext,
                )
                # print(out_frame_path)
                torchvision.utils.save_image(
                    processed_output,
                    os.path.join(self.temp_folder_path, out_frame_path),
                )

    def process_frames(self, left_frame_path_first=True):
        start_time = time.time()
        temp_folder_list = sorted(os.listdir(self.temp_folder_path))
        for L_frame, R_frame in zip(
            self.L_frame_list, tqdm(temp_folder_list, total=len(temp_folder_list))
        ):
            if left_frame_path_first:
                L_path = os.path.join(self.L_frame_path, L_frame)
                R_path = os.path.join(self.temp_folder_path, R_frame)
            else:
                L_path = os.path.join(self.temp_folder_path, R_frame)
                R_path = os.path.join(self.L_frame_path, L_frame)
            # print(L_path)
            # print(R_path)
            in_frame = Image.open(L_path)
            out_frame = Image.open(R_path)

            LR_frame = Image.new("RGB", (2 * self.out_width, self.out_height))
            LR_frame.paste(in_frame, (0, 0))

            LR_frame.paste(out_frame, (self.out_width, 0))

            LR_frame.save(os.path.join(self.LR_folder_path, L_frame.replace("L", "LR")))
            self.process_all_images_time += time.time() - start_time
            in_frame.close()
            out_frame.close()
        if self.delete_temp:
            print("Removing the temp folder")
            shutil.rmtree(self.temp_folder_path)

    def show_info(self):
        print("Time taken to process all the L frames: ", self.process_all_images_time)
        print("All frames processed and saved in the output folder")
        print("Output folder: ", self.LR_folder_path)


class VAE_model_to_frames(model_to_frames):
    def __init__(
        self,
        input_folder_path,
        output_folder_path,
        out_width,
        out_height,
        model,
        patch_list,
        frame_ext=".jpg",
        L_frame_path=None,
        model_parameters_path=None,
        delete_temp=False,
        device="cuda",
    ):
        super().__init__(
            input_folder_path,
            output_folder_path,
            out_width,
            out_height,
            model,
            frame_ext,
            L_frame_path,
            model_parameters_path,
            delete_temp,
            device,
        )
        self.patch_list = patch_list

    def process_model(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        return self.model

    def process_input_for_model(self, in_path):
        return torch.load(in_path).to(self.device)

    def model_output(self, processed_input):
        with torch.no_grad():
            out = latent_to_frame(
                image=processed_input,
                patch_size_list=self.patch_list,
                model=self.processed_model,
                upsample=True,
                factor=8,
                is_tensor=True,
            ).get_result()
        return out

    def process_model_output(self, raw_model_output):
        return raw_model_output
