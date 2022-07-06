## general imports
import random
import numpy as np
from typing import Tuple
## import torch related functions
import torch
from torch import Tensor
## import torchvision functions
import torchvision.transforms.functional as F


__all__ = [
    "image2tensor", "tensor2image",
    "unnormalize", "normalize",
    "random_vertically_flip", "random_horizontally_flip",
    "random_crop", "center_crop",
    "random_adjust_brightness", "random_adjust_contrast",
    "random_rotate"

]

def image2tensor(image: np.ndarray) -> Tensor:
    image = F.to_tensor(image)
    return image

def tensor2image(tensor: Tensor) -> np.ndarray:
    tensor = F.to_pil_image(tensor)
    return tensor

def unnormalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    image = image * 255.0
    return  image

def normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    image = image / 255.0
    return image

def random_vertically_flip(lr: np.ndarray, hr: np.ndarray, p=0.5) -> Tuple[np.ndarray, np.ndarray]:

    if torch.rand(1).item() > p:
        hr = F.vflip(hr)
        lr = F.vflip(lr)
    ## return lr and hr  
    return lr, hr

def random_horizontally_flip(lr: np.ndarray, hr: np.ndarray, p=0.5) -> Tuple[np.ndarray, np.ndarray]:

    if torch.rand(1).item() > p:
        hr = F.hflip(hr)
        lr = F.hflip(lr)
    ## return lr and hr    
    return lr, hr

def random_crop(lr: np.ndarray, hr: np.ndarray, image_size: int, upscale_factor: int) -> Tuple[np.ndarray, np.ndarray]:
    ## random crop on images -  
    w, h = hr.size
    left = torch.randint(0, w - image_size + 1, size=(1,)).item()
    top = torch.randint(0, h - image_size + 1, size=(1,)).item()
    
    bottom = top + image_size
    right = left + image_size

    lr = lr.crop((left // upscale_factor,
                  top // upscale_factor,
                  right // upscale_factor,
                  bottom // upscale_factor))
    hr = hr.crop((left, top, right, bottom))

    return lr, hr

def center_crop(lr: np.ndarray, hr: np.ndarray, image_size: int, upscale_factor: int) -> Tuple[np.ndarray, np.ndarray]:
    ## cropping center
    w, h = hr.size
    top = (h - image_size) // 2
    left = (w - image_size) // 2
    
    right = left + image_size
    bottom = top + image_size

    lr = lr.crop((left // upscale_factor,
                  top // upscale_factor,
                  right // upscale_factor,
                  bottom // upscale_factor))
    hr = hr.crop((left, top, right, bottom))

    return lr, hr


def random_adjust_brightness(lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ## adjust the brightness - augmentation 
    factor = random.uniform(0.25, 4)
    hr = F.adjust_brightness(hr, factor)
    lr = F.adjust_brightness(lr, factor)
    

    return lr, hr

def random_adjust_contrast(lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:


    factor = random.uniform(0.25, 4)
    hr = F.adjust_contrast(hr, factor)
    lr = F.adjust_contrast(lr, factor)
    

    return lr, hr

def random_rotate(lr: np.ndarray, hr: np.ndarray, degrees: int) -> Tuple[np.ndarray, np.ndarray]:

    degrees = random.choice((+degrees, -degrees))
    hr = F.rotate(hr, degrees)
    lr = F.rotate(lr, degrees)
    

    return lr, hr

