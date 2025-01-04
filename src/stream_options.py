### importing standard libraries
from tqdm import tqdm
import os
import torch
import dask
import glob
from dask.diagnostics import ProgressBar  # type: ignore
import shutil
from PIL import Image
import time

### importing user defined libraries
from src.utils.file_utils import make_folder, filenames
from src.model_to_latents import save_tensor_as_png_all
from src.frames_to_video import latents_to_video_ffmpeg
from src.utils.file_utils import make_folder, filenames
from src.utils.frame_utils import get_frame_dim


class choose_stream_options:
    def __init__(
        self,
        latent_tensors_path,
        video_name,
        is_top_down,
        img_format,
        video_ext,
        output_folder_path=None,
        num_frames=[None, None],
    ):
        self.latent_tensors_path = latent_tensors_path
        self.output_folder_path = output_folder_path
        self.video_name = video_name
        self.img_format = img_format
        self.is_top_down = is_top_down
        if is_top_down:
            self.ext_1 = "T" + img_format
            self.ext_2 = "B" + img_format
        else:
            self.ext_1 = "L" + img_format
            self.ext_2 = "R" + img_format

        self.video_ext = video_ext

        _, _, self.base_folder = filenames(self.latent_tensors_path)
        if self.output_folder_path != None:
            self.base_folder = self.output_folder_path

        self.num_frames = num_frames
        

    #save diff tensors from the folder
    def save_diff_tensor(self, res_ext):
        folder_name = "diff_frames"

        latent_png_path = save_tensor_as_png_all(
            self.latent_tensors_path,
            None,
            folder_name,
            img_format=self.img_format,
            save_residual_ext=res_ext,
            extra_pixel=None,
            return_values = True,
            num_images=self.num_frames,
        )
        return latent_png_path

    #creating two temp folders to store the frames in seperate option
    def seperate_folders(self, out_folder_path, folder_1_name, folder_2_name):
        print("\n Creating a temporary folder to store the frames ...\n")
        temp_folder_ext_1 = make_folder(
            folder_name=folder_1_name, folder_path=out_folder_path, remove_if_exists=True
        )

        temp_folder_ext_2 = make_folder(
            folder_name=folder_2_name, folder_path=out_folder_path, remove_if_exists=True
        )

        return temp_folder_ext_1, temp_folder_ext_2

    #save tensors from the folder to temp folder with diff 
    def seperate_residual_logic(self, latent_png_path, temp_folder_ext_1, temp_folder_ext_2):
        for frame in sorted(os.listdir(latent_png_path)):
            if frame.endswith(self.ext_1):
                shutil.copy2(
                    os.path.join(latent_png_path, frame),
                    temp_folder_ext_1,
                )
                if os.path.exists(
                    os.path.join(
                        latent_png_path,
                        frame.replace(
                            self.ext_1,
                            self.ext_2.replace(
                                self.img_format, "_diff" + self.img_format
                            ),
                        ),
                    )
                ):
                    shutil.copy2(
                        os.path.join(
                            latent_png_path,
                            frame.replace(
                                self.ext_1,
                                self.ext_2.replace(
                                    self.img_format, "_diff" + self.img_format
                                ),
                            ),
                        ),
                        temp_folder_ext_2,
                    )

        print("Seperate Frames copied to a temporary folder successfully!")
        print(f" Location: {temp_folder_ext_1} and {temp_folder_ext_2}")

    def seperate_raw_logic(self, latent_png_path, temp_folder_ext_1, temp_folder_ext_2):
        for frame in sorted(os.listdir(latent_png_path)):
            #print(frame)
            if frame.endswith(self.ext_1):
                shutil.copy2(
                    os.path.join(latent_png_path, frame),
                    temp_folder_ext_1,
                )
                if os.path.exists(
                    os.path.join(
                        latent_png_path,
                        frame.replace(
                            self.ext_1,
                            self.ext_2,
                        ),
                    )
                ):
                    shutil.copy2(
                        os.path.join(
                            latent_png_path,
                            frame.replace(
                                self.ext_1,
                                self.ext_2,
                            ),
                        ),
                        temp_folder_ext_2,
                    )
        print("Raw Frames copied to a temporary folder successfully!")
        print(f" Location: {temp_folder_ext_1} and {temp_folder_ext_2}")


    def seperate(self, option, overwrite):

        print(f"\n ----------- Selected Stream Option: Seperate {option} ----------- \n")

        if option == "residual":
            latent_png_path = self.save_diff_tensor(self.ext_1.replace(self.img_format, ""))
            #print(os.listdir(latent_png_path))
            _, _, base_folder = filenames(latent_png_path)

            temp_1_name = "sim_Lat"
            temp_2_name = "sim_Res"

            temp_folder_ext_1, temp_folder_ext_2 = self.seperate_folders(base_folder, temp_1_name, temp_2_name)

            self.seperate_residual_logic(latent_png_path, temp_folder_ext_1, temp_folder_ext_2)

        if option == "raw":
            latent_png_path = self.save_diff_tensor(None)
            #print(os.listdir(latent_png_path))

            _, _, base_folder = filenames(latent_png_path)

            temp_1_name = "sim_raw_"+self.ext_1.replace(self.img_format, "")
            temp_2_name = "sim_raw_"+self.ext_2.replace(self.img_format, "")

            temp_folder_ext_1, temp_folder_ext_2 = self.seperate_folders(base_folder, temp_1_name, temp_2_name)

            self.seperate_raw_logic(latent_png_path, temp_folder_ext_1, temp_folder_ext_2)

        print("\n------------- Creating video from the frames ... -------------\n")

        output_video_name = os.path.join(self.base_folder, self.video_name + "_"+temp_1_name)

        latents_to_video_ffmpeg(
            temp_folder_ext_1,
            output_video_name,
            self.video_ext,
            in_fps=30,
            img_format=self.img_format,
            num_frames=[None, None],
            out_folder_path=self.output_folder_path,
            overwrite_previous_video=overwrite,
        )

        output_video_name = os.path.join(self.base_folder, self.video_name + "_"+temp_2_name)

        latents_to_video_ffmpeg(
            temp_folder_ext_2,
            output_video_name,
            self.video_ext,
            in_fps=30,
            img_format=self.img_format,
            num_frames=[None, None],
            out_folder_path=self.output_folder_path,
            overwrite_previous_video=True,
        )

        # delete the temp folders
        print("\n --------- Deleting temporary folders --------- \n")
        shutil.rmtree(temp_folder_ext_1)
        shutil.rmtree(temp_folder_ext_2)

################# alternate option ####################
    def alternate_residual_logic(self, latent_png_path, temp_folder_ext):
        flag = 0
        for frame in sorted(os.listdir(latent_png_path)):

            if frame.endswith(self.ext_1):
                shutil.copy2(
                    os.path.join(latent_png_path, frame),
                    temp_folder_ext,
                )
                os.rename(
                    os.path.join(temp_folder_ext, frame),
                    os.path.join(temp_folder_ext, f"a_{flag:06d}" + self.img_format),
                )
                flag += 1
        num_latents_till_now = len(os.listdir(temp_folder_ext))


        for frame in sorted(os.listdir(latent_png_path)):
            if frame.endswith(self.ext_1):
                if os.path.exists(
                    os.path.join(
                        latent_png_path,
                        frame.replace(
                            self.ext_1,
                            self.ext_2.replace(
                                self.img_format, "_diff" + self.img_format
                            ),
                        ),
                    )
                ):
                    shutil.copy2(
                        os.path.join(
                            latent_png_path,
                            frame.replace(
                                self.ext_1,
                                self.ext_2.replace(
                                    self.img_format, "_diff" + self.img_format
                                ),
                            ),
                        ),
                        temp_folder_ext,
                    )
                    os.rename(
                        os.path.join(
                            temp_folder_ext,
                            frame.replace(
                                self.ext_1,
                                self.ext_2.replace(
                                    self.img_format, "_diff" + self.img_format
                                ),
                            ),
                        ),
                        os.path.join(
                            temp_folder_ext,
                            f"a_{num_latents_till_now:06d}" + self.img_format,
                        ),
                    )
                    num_latents_till_now += 1


    def alternate_raw_logic(self, latent_png_path, temp_folder_ext):
        flag = 0
        for frame in sorted(os.listdir(latent_png_path)):
            if frame.endswith(self.ext_1):
                shutil.copy2(
                    os.path.join(latent_png_path, frame),
                    temp_folder_ext,
                )
                os.rename(
                    os.path.join(temp_folder_ext, frame),
                    os.path.join(temp_folder_ext, f"a_{flag:06d}" + self.img_format),
                )
                flag += 1
        num_latents_till_now = len(os.listdir(temp_folder_ext))

        for frame in sorted(os.listdir(latent_png_path)):
            if frame.endswith(self.ext_1):
                if os.path.exists(
                    os.path.join(
                        latent_png_path,
                        frame.replace(
                            self.ext_1,
                            self.ext_2,
                        ),
                    )
                ):
                    shutil.copy2(
                        os.path.join(
                            latent_png_path,
                            frame.replace(
                                self.ext_1,
                                self.ext_2,
                            ),
                        ),
                        temp_folder_ext,
                    )
                    os.rename(
                        os.path.join(
                            temp_folder_ext,
                            frame.replace(
                                self.ext_1,
                                self.ext_2,
                            ),
                        ),
                        os.path.join(
                            temp_folder_ext,
                            f"a_{num_latents_till_now:06d}" + self.img_format,
                        ),
                    )
                    num_latents_till_now += 1


    def alternate(self, option, overwrite):
        print(f"\n ----------- Selected Stream Option: Alternate {option} ----------- \n")
        

        if option == "residual":
            latent_png_path = self.save_diff_tensor(self.ext_1.replace(self.img_format, ""))

            _, _, base_folder = filenames(latent_png_path)

            print("\n Creating a temporary folder to store the frames ...")

            temp_folder_ext = make_folder(
                folder_name="f_pack_res", folder_path=base_folder, remove_if_exists=True
            )

            self.alternate_residual_logic(latent_png_path, temp_folder_ext)

        if option == "raw":
            latent_png_path = self.save_diff_tensor(None)

            _, _, base_folder = filenames(latent_png_path)

            print("\n Creating a temporary folder to store the frames ...")

            temp_folder_ext = make_folder(
                folder_name="f_pack_raw", folder_path=base_folder, remove_if_exists=True
            )
            self.alternate_raw_logic(latent_png_path, temp_folder_ext)

        print(
            "\n--------- Frames copied to a temporary folder successfully! ---------\n"
        )

        print("\n------------- Creating video from the frames ... -------------\n")

        output_video_name = os.path.join(self.base_folder, self.video_name + "_alt"+ option[:3])
        latents_to_video_ffmpeg(
            temp_folder_ext,
            output_video_name,
            self.video_ext,
            in_fps=30,
            img_format=self.img_format,
            num_frames=[None, None],
            out_folder_path=self.output_folder_path,
            overwrite_previous_video=overwrite,
        )
        #delete the temp folders
        print("\n --------- Deleted temporary folders --------- \n")
        shutil.rmtree(temp_folder_ext)


################# merge option ####################

    @dask.delayed  # type: ignore
    def process_one_frame(
        self, image_pair, top_down, number, destinationFolder, img_format, frame_info
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

    def merge_residual_logic(self, latent_png_path, destinationFolder):
        number = 1
        delayed_tasks = []
        frame_info = get_frame_dim(
            sorted(glob.glob(latent_png_path + "/*" + self.img_format))[0], is_tensor=False
        )
        for frame in tqdm(sorted(os.listdir(latent_png_path)), desc= "Appending dask tasks...", total=len(os.listdir(latent_png_path))):
            if frame.endswith(self.ext_1):
                ext_1_frame_path = os.path.join(latent_png_path, frame)
                ext_2_frame_path = os.path.join(
                    latent_png_path,
                    frame.replace(self.ext_1, self.ext_2.replace(self.img_format, "_diff" + self.img_format)),
                )
                if os.path.exists(ext_2_frame_path):
                    frame_pair = [ext_1_frame_path, ext_2_frame_path]
                    delayed_tasks.append(
                        self.process_one_frame(
                            frame_pair,
                            self.is_top_down,
                            number,
                            destinationFolder,
                            self.img_format,
                            frame_info,
                        )
                    )
                    number += 1

        return delayed_tasks
            

    def merge_raw_logic(self, latent_png_path, destinationFolder):
        number = 1
        delayed_tasks = []
        frame_info = get_frame_dim(
            sorted(glob.glob(latent_png_path + "/*" + self.img_format))[0], is_tensor=False
        )
        for frame in tqdm(sorted(os.listdir(latent_png_path)), desc= "Appending dask tasks...", total=len(os.listdir(latent_png_path))):
            if frame.endswith(self.ext_1):
                ext_1_frame_path = os.path.join(latent_png_path, frame)
                ext_2_frame_path = os.path.join(
                    latent_png_path,
                    frame.replace(self.ext_1, self.ext_2),
                )
                if os.path.exists(ext_2_frame_path):
                    frame_pair = [ext_1_frame_path, ext_2_frame_path]
                    delayed_tasks.append(
                        self.process_one_frame(
                            frame_pair,
                            self.is_top_down,
                            number,
                            destinationFolder,
                            self.img_format,
                            frame_info,
                        )
                    )
                    number += 1
        
        return delayed_tasks
        

    

    def merge(self, option, overwrite):

        print(f"\n ----------- Selected Stream Option: Merge {option} ----------- \n")

        if option == "residual":
            latent_png_path = self.save_diff_tensor(self.ext_1.replace(self.img_format, ""))

            _, _, base_folder = filenames(latent_png_path)

            print("\n Creating a temporary folder to store the frames ...")

            temp_folder_ext = make_folder(
                folder_name="f_comp_res", folder_path=base_folder, remove_if_exists=True
            )
        
            
            delayed_tasks = self.merge_residual_logic(latent_png_path, temp_folder_ext)
        
        if option == "raw":
            latent_png_path = self.save_diff_tensor(None)

            _, _, base_folder = filenames(latent_png_path)

            print("\n Creating a temporary folder to store the frames ...")

            temp_folder_ext = make_folder(
                folder_name="f_comp_raw", folder_path=base_folder, remove_if_exists=True
            )
        
            delayed_tasks = self.merge_raw_logic(latent_png_path, temp_folder_ext)
        
        print("\n --------- Processing the dasks tasks--------- \n")
        with ProgressBar():
            dask.compute(*delayed_tasks) # type: ignore
        
        print("\n--------- Frames copied to a temporary folder successfully! ---------\n")

        print("\n------------- Creating video from the frames ... -------------\n")

        output_video_name = os.path.join(self.base_folder, self.video_name +option[:3])

        latents_to_video_ffmpeg(
            temp_folder_ext,
            output_video_name,
            self.video_ext,
            in_fps=30,
            img_format=self.img_format,
            num_frames=[None, None],
            out_folder_path=self.output_folder_path,
            overwrite_previous_video=overwrite,
        )

        # delete the temp folders
        print("\n --------- Deleted temporary folders --------- \n")
        shutil.rmtree(temp_folder_ext)



            
        
