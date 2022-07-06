import os
from typing import Tuple
from PIL import Image


## importing torch utilities
from torch import Tensor
from torch.utils.data import Dataset
## importing torchvision related functionalities
from torchvision.transforms.functional import InterpolationMode as IMode
import torchvision.transforms as transforms

## using the utility functions defined in the img_proc_util
from img_proc_util import image2tensor
from img_proc_util import center_crop
from img_proc_util import random_horizontally_flip
from img_proc_util import random_rotate
from img_proc_util import random_crop



__all__ = [ "CustomDataset","BaseDataset"]

## defining custom dataset
## it will handle the path for all the images
## get the upscaling factor and image sizes
## it will apply data augmentation techniques such as
## crop rotate and horizontal flip
## finally convert to tensor
class CustomDataset(Dataset):

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(CustomDataset, self).__init__()
        upscale_factor = int(upscale_factor)
        tmp_hr_dir_path = os.path.join(dataroot, f"HR")
        tmp_lr_dir_path = os.path.join(dataroot, f"LRunknownx{upscale_factor}")
        
        self.mode = mode
        self.filenames = os.listdir(tmp_lr_dir_path)
        self.tmp_lr_file_names = [ os.path.join(tmp_lr_dir_path, x) for x in self.filenames ]
        self.tmp_hr_file_names = [os.path.join(tmp_hr_dir_path, x) for x in self.filenames]
        self.upscale_factor = upscale_factor
        self.image_size = image_size  
        
        

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        lr = Image.open(self.tmp_lr_file_names[index])
        hr = Image.open(self.tmp_hr_file_names[index])

        if self.mode == "train":
            lr, hr = random_crop(lr, hr, self.image_size, self.upscale_factor)
            lr, hr = random_rotate(lr, hr, 90)
            lr, hr = random_horizontally_flip(lr, hr, 0.5)
        else:
            lr, hr = center_crop(lr, hr, self.image_size, self.upscale_factor)

        lr = image2tensor(lr)
        hr = image2tensor(hr)

        return lr, hr

    def __len__(self) -> int:
        return len(self.filenames)


## base dataset class extending the dataset class from pytorch
## applies augmentation techniques such as random crop, rotation
## horizontal flip and tensor
## resizing and center crop is also used
## final conversion to tensor
class BaseDataset(Dataset):


    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(BaseDataset, self).__init__()
        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
        lr_img_size = (image_size // upscale_factor, image_size // upscale_factor)
        hr_img_size = (image_size, image_size)
        
        
        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(hr_img_size),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
            ])
        else:
            self.hr_transforms = transforms.Compose([
                transforms.CenterCrop(hr_img_size),
                transforms.ToTensor()
            ])
        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(lr_img_size, interpolation=IMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        hr = Image.open(self.filenames[index])
        temp_lr = self.lr_transforms(hr)
        temp_hr = self.hr_transforms(hr)
        

        return temp_lr, temp_hr

    def __len__(self) -> int:
        return len(self.filenames)