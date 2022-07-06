import shutil
import warnings
from typing import Tuple
import numpy as np
import cv2

from PIL import Image
from config import *
from img_proc_util import *
##  sklearn image related functions

import skimage.color
import skimage.io
import skimage.metrics
from skimage import img_as_ubyte
## importing torch and torchvision utilities
import torch
import torchvision.utils


def image_quality_assessment(sr_path: str, hr_path: str) -> Tuple[float, float, float]:

    ## reading hr and sr image
    hr_image = skimage.io.imread(hr_path)
    sr_image = skimage.io.imread(sr_path)
    

    if sr_image.shape[1] != sr_image.shape[0]:
        warnings.warn("Image shape - height and width not equal")
    if hr_image.shape != sr_image.shape:
        warnings.warn("spectrum calculation error")
    
    spectrum = cal_spectrum(sr_image, hr_image)
    ## calculation of PSNR and SSIM
    psnr, ssim = cal_psnr_and_ssim(sr_image, hr_image)
    
    return psnr, ssim, spectrum

def cal_spectrum(sr_image, hr_image) -> float:

    sr_temp = img_as_ubyte(sr_image)
    hr_temp = img_as_ubyte(hr_image)
    sr = cv2.cvtColor(sr_temp, cv2.COLOR_RGB2GRAY)
    hr = cv2.cvtColor(hr_temp, cv2.COLOR_RGB2GRAY)

    sr_shape = sr.shape[0]
    total_hist_sr = []
    total_hist_hr = []
    for hist_height in range(sr_shape):
 
        hist_sr = cv2.calcHist([sr[hist_height, :]], [0], None, [sr_shape], [0, 255])
        hist_hr = cv2.calcHist([hr[hist_height, :]], [0], None, [sr_shape], [0, 255])
        total_hist_sr.append(hist_sr)
        total_hist_hr.append(hist_hr)


    total_spectrum_sr = []
    total_spectrum_hr = []
    for idx in range(sr_shape):

        fft_sr = np.fft.fft(total_hist_sr[idx])
        fft_hr = np.fft.fft(total_hist_hr[idx])

        spectrum_sr = np.abs(fft_sr)
        spectrum_hr = np.abs(fft_hr)

        spectrum_sr = spectrum_sr[range(sr_shape // 2)]
        spectrum_hr = spectrum_hr[range(sr_shape // 2)]
        total_spectrum_sr.append(spectrum_sr)
        total_spectrum_hr.append(spectrum_hr)


    average_spectrum_sr = []
    average_spectrum_hr = []
    diff = 0.
    for spectrum_val in range(sr_shape // 2):
        total_spectrum_sr = 0
        total_spectrum_hr = 0
        for idx in range(sr_shape):
            total_spectrum_sr += total_spectrum_sr[idx][spectrum_val]
            total_spectrum_hr += total_spectrum_hr[idx][spectrum_val]
        average_spectrum_sr.append(total_spectrum_sr / sr_shape)
        average_spectrum_hr.append(total_spectrum_hr / sr_shape)


    
    for index in range(sr_shape // 2):
        diff += (average_spectrum_hr[index] - average_spectrum_sr[index]) ** 2

    spectrum_val = float(np.sqrt(diff / (sr_shape / 2)))

    return spectrum_val

def cal_psnr_and_ssim(sr_image, hr_image) -> Tuple[float, float]:
    ## calculate psnr and ssim 

    sr_temp = normalize(sr_image)
    hr_temp = normalize(hr_image)

    sr_temp = skimage.color.rgb2ycbcr(sr_temp)[:, :, 0:1]
    hr_temp = skimage.color.rgb2ycbcr(hr_temp)[:, :, 0:1]

    sr = normalize(sr_temp)
    hr = normalize(hr_temp)
    ssim = skimage.metrics.structural_similarity(sr,
                                                 hr,
                                                 win_size=11,
                                                 gaussian_weights=True,
                                                 multichannel=True,
                                                 data_range=1.0,
                                                 K1=0.01,
                                                 K2=0.03,
                                                 sigma=1.5)
    psnr = skimage.metrics.peak_signal_noise_ratio(sr, hr, data_range=1.0)
    
    return psnr, ssim




def main() -> None:
    ## defining the main function to get the functions in line

    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    ## load models to evaluate
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    model.half()

    total_spectrum_val = 0.0
    total_ssim_val = 0.0
    total_psnr_val = 0.0
    
    


    filenames = os.listdir(lr_dir)

    total_files = len(filenames)

    for index in range(total_files):
        lr_path = os.path.join(lr_dir, filenames[index])
        sr_path = os.path.join(sr_dir, filenames[index])
        hr_path = os.path.join(hr_dir, filenames[index])

        lr_image = Image.open(lr_path)
        lr_tensor = image2tensor(lr_image).to(device).unsqueeze(0)
        lr_tensor = lr_tensor.half()
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
            torchvision.utils.save_image(sr_tensor, sr_path)


        print(f"Processing `{os.path.abspath(lr_path)}`...")
        psnr, ssim, spectrum = image_quality_assessment(sr_path, hr_path)
        total_spectrum_val += spectrum
        
        total_ssim_val += ssim
        total_psnr_val += psnr
        

    print(f"PSNR:     {total_psnr_val / total_files:.2f}.\n"
          f"SSIM:     {total_ssim_val / total_files:.4f}.\n"
          f"Spectrum: {total_spectrum_val / total_files:.6f}.\n")


if __name__ == "__main__":
    main()
