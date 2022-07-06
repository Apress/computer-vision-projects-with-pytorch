import os
import argparse
import cv2
from PIL import Image
from model_srgan import Generator
## import torch relates features
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision as tv
import torchvision.transforms as transforms
## Skimage for processing and transform
import skimage.io
import skimage.transform
import skimage.data
import skimage.transform
## generic imports
import numpy as np
import matplotlib.pyplot as plt


class FeatureExtractorImg(nn.Module):
    def __init__(self, sub_module, extracted_layers):
        super(FeatureExtractorImg, self).__init__()
        self.sub_module = sub_module
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = {}
        for l_name, module in self.sub_module._modules.items():

            x = module(x)
            if self.extracted_layers is None or l_name in self.extracted_layers and 'fc' not in l_name:
                outputs[l_name] = x

        return outputs

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def get_image(img_name, transform):
    ## Transformation
    image = skimage.io.imread(img_name)
    ## transform
    image = skimage.transform.resize(image, (256, 256))
    ## array conversion
    image = np.asarray(image, dtype=np.float32)
    return transform(image)


def get_feature():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pic_dir = 'lr_024067.bmp'
    ## convert to tensors
    transform = transforms.ToTensor()
    img = get_image(pic_dir, transform)
    

    img = img.unsqueeze(0)
    img = img.to(device)
    exact_list = None
    net = Generator().to(device)
    net.load_state_dict(torch.load('20210728.pth'))
    
    dst = './feautures'
    thrd_dim = 64

    myexactor = FeatureExtractorImg(net, exact_list)
    outs = myexactor(img)
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            ds_path = os.path.join(dst, k)
            make_dirs(ds_path)

            feature = features.data.cpu().numpy()
            feature_img = feature[i,:,:]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
            ## apply color map 
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < thrd_dim:
                ## writing temp files
                tmp_file = os.path.join(ds_path, str(i) + '_' + str(thrd_dim) + '.png')
                temp_img = feature_img.copy()
                temp_img = cv2.resize(temp_img, (thrd_dim,thrd_dim), interpolation =  cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, temp_img)
            
            dst_file = os.path.join(ds_path, str(i) + '.png')
            cv2.imwrite(dst_file, feature_img)

if __name__ == '__main__':
    get_feature()


