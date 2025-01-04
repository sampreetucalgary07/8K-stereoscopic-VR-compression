# from src.utils.tensor_utils import image_to_tensor
from PIL import Image


def checknumpyImage(image, numpyImage=False):
    if numpyImage:
        image = image.permute(2, 1, 0).cpu().detach().numpy()
    else:
        image = image.permute(2, 1, 0).cpu()

    return image


# function to return the shape of the frame
def get_frame_dim(frame, is_tensor=False, device="cuda"):
    if is_tensor:
        return {
            "channels": frame.shape[-3],
            "height": frame.shape[-2],
            "width": frame.shape[-1],
        }
    else:
        img = Image.open(frame)
        img.close()
        return {
            "channels": img.mode,
            "height": img.size[1],
            "width": img.size[0],
        }
