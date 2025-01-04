import os
import shutil
from IPython.display import clear_output
from tqdm import tqdm
import pickle
from PIL import Image

############################ Functions for Data Preprocessing ############################


def filenames(mp4_file_path):
    """
    function to make filenames
    """
    # Extract the filename: 'voyager.mp4'
    basename = os.path.basename(mp4_file_path)
    base_folder = os.path.dirname(mp4_file_path)
    video_name, video_ext = os.path.splitext(basename)  # Split the extension: 'voyager'
    return (
        video_name,
        video_ext,
        base_folder,
    )


def make_folder(folder_name, folder_path, remove_if_exists=False):
    """
    Function to make a new folder (if not there)
    """

    folder = os.path.join(folder_path, folder_name)

    try:
        if os.path.exists(folder):
            if remove_if_exists:
                shutil.rmtree(folder)
                print(f"\nExisting folder ({folder_name}) deleted successfully!")
                os.makedirs(folder)
                print(f"\n({folder_name}) named folder created successfully!")
                print("at (Folder path): ", folder)
            else:
                raise ValueError(
                    f"{folder_name} Folder already exists! "
                    "Set the parameter remove_if_exists=True to remove the folder and create a new one!"
                )

        else:
            os.makedirs(folder)
            print(f"\n{folder_name} folder created successfully!")
            print("at (Folder path): ", folder)

    except ValueError as e:
        print(e)

    return f"{folder}"


def crop_and_copy_images(width, height, source_folder, destination_folder=None):

    # Function to crop and copy images from the source folder to the destination folder

    # Create the destination folder if it doesn't exist
    folder_name, _, base_folder = filenames(source_folder)

    if destination_folder is not None:
        base_folder = destination_folder

    destination_folder_path = make_folder(
        folder_name + "_" + str(width) + "x" + str(height),
        base_folder,
        remove_if_exists=True,
    )

    # Get the list of files in the source folder
    files = os.listdir(source_folder)

    # Crop and copy each image to the destination folder
    for file in tqdm(files, total=len(files), desc="Cropping images ..."):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(source_folder, file)
            image = Image.open(image_path)

            # Resize the image to the specified size
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
            # cropped_image = image.crop((0, 0, crop_size[0], crop_size[1]))

            # Save the cropped image to the destination folder
            destination_path = os.path.join(destination_folder_path, file)
            resized_image.save(destination_path)

    print("Images cropped and copied successfully.")
    print("Images saved at : ", destination_folder_path)


def check_image_size(folder_path, return_img_size=False):

    # Function to check the size of images in the folder

    image_sizes = set()

    for filename in tqdm(os.listdir(folder_path), desc="Checking image sizes ..."):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image_sizes.add(image.size)

    if len(image_sizes) == 1:
        print("Folder path: ", folder_path)
        print("All frames have the same size.")
        print("Single frame size: {}".format(list(image_sizes)[0]))
        print("No. of frames in the folder : ", len(os.listdir(folder_path)))
        if return_img_size:
            return list(image_sizes)[0]  # Return the image size

    else:
        raise ValueError("Images have different sizes.")


def save_dict_to_file(dictionary, file_name, file_path):
    # save dict as pickle file
    filePath = os.path.join(file_path, file_name)
    with open(filePath, "wb") as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dictionary saved to {filePath}")


def load_dict_from_file(dict_path):
    # load dict from pickle file
    with open(dict_path, "rb") as handle:
        loaded_dict = pickle.load(handle)

    print(f"Dictionary loaded from {dict_path}")
    return loaded_dict


def save_dict_to_pickle(dict_, path):
    with open(path, "wb") as file:
        pickle.dump(dict_, file)


def load_dict_from_pickle(path):
    with open(path, "rb") as file:
        dict_ = pickle.load(file)
    return dict_


def delete_images_with_prefix(folder_path, prefix):
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)


def copy_images(source_folder, destination_folder, num_images=300):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all image files in the source folder
    image_files = sorted(os.listdir(source_folder))

    # Copy and paste the specified number of images
    for i in range(num_images):
        # Check if there are no more images to copy
        if i >= len(image_files):
            print(
                f"Only {len(image_files)} images found in the source folder. Cannot copy {num_images} images."
            )
            break

        # Get the source and destination paths for the image
        source_path = os.path.join(source_folder, image_files[i])
        destination_path = os.path.join(destination_folder, image_files[i])

        # Copy the image to the destination folder
        shutil.copyfile(source_path, destination_path)

    print(
        f"Copied {min(num_images, len(image_files))} images from {source_folder} to {destination_folder}."
    )


def merge_folders(source1, source2, destination):
    # Ensure the destination folder exists
    os.makedirs(destination, exist_ok=True)

    # Copy contents of source1 to destination
    for item in os.listdir(source1):
        source_path = os.path.join(source1, item)
        destination_path = os.path.join(destination, item)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)

    # Copy contents of source2 to destination
    for item in os.listdir(source2):
        source_path = os.path.join(source2, item)
        destination_path = os.path.join(destination, item)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)

    # Example usage:
    # source_folder1 = "/path/to/source1"
    # source_folder2 = "/path/to/source2"
    # destination_folder = "/path/to/destination"

    # merge_folders(source_folder1, source_folder2, destination_folder)


def get_image_list(image_list, num_list=[0, -1]):

    sorted_list = sorted(os.listdir(image_list))[num_list[0] : num_list[1]]
    print("Number of images in the directory: ", len(sorted_list))
    return [os.path.join(image_list, img) for img in sorted_list]


def file_size(file_path, print_value=True, return_value=False):
    if print_value:
        print(f"File size : {os.path.getsize(file_path) / (1024 * 1024)} MB")
    if return_value:
        return os.path.getsize(file_path) / (1024 * 1024)
