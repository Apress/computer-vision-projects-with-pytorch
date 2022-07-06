from PIL import Image
from torch.utils.data import Dataset
import os
## basic imports
import os
import numpy as np
## torch imports
import torch
import torch.utils.data
from torch.utils.data import Dataset
## torchvision imports
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
## image utilities
from PIL import Image
import matplotlib.pyplot as plt
import cv2
## code utilities
import random
import warnings
warnings.filterwarnings('ignore')

def generate_prediction(image_path, conf):
    ## helper function to generate predictions
    image = Image.open(image_path)
    transform = T.Compose([T.ToTensor()])
    image = transform(image)

    image = image.to(device)
    predicted = final_model([image])
    predicted_score = list(predicted[0]['scores'].detach().cpu().numpy())
    predicted_temp = [predicted_score.index(x) for x in predicted_score if x>conf][-1]
    masks = (predicted[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max())
    predicted_class_val = [CLASSES[i] for i in list(predicted[0]['labels'].cpu().numpy())]
    predicted_box_val = [[(i[0], i[1]), (i[2], i[3])] for i in list(predicted[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:predicted_temp+1]
    predicted_class_name = predicted_class_val[:predicted_temp+1]
    predicted_box_score = predicted_box_val[:predicted_temp+1]
    
    return masks, predicted_box_score, predicted_class_name