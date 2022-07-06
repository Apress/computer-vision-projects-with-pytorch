import os
## importing torch utilities
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
## optimiser
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter
## import from model
from model_srgan import Generator
from model_srgan import ContentLoss
from model_srgan import Discriminator


## config for train and validation

torch.manual_seed(111)  
device = torch.device("cuda:0")                      
upscale_factor = 4                       
mode  = "train"   
cudnn.benchmark  = True                   
               
exp_name         = "exp000"               

## configure for train pipeline 
if mode == "train":
    batch_size  = 16 
    # dataset conf
    train_dir  = "data/ImageNet/train"               
    image_size  = 96
    valid_dir = "data/Ima geNet/valid"                           
                             

    # model conf
    generator = Generator().to(device) 
    discriminator = Discriminator().to(device)  
         

    # train - for pretrained model
    resume  = False 
    start_epoch = 0 ## same as above
    start_p_epoch = 0 ## init as per choice                          
    resume_g_weight = ""
    resume_d_weight = ""                                               
    resume_p_weight = ""                         
                            
                              

    # epochs - CONV block and GAN
    epochs  = 5
    p_epochs  = 20                         
                              

    # loss function to device
    content_criterion     = ContentLoss().to(device)
    psnr_criterion        = nn.MSELoss().to(device)
    adversarial_criterion = nn.BCELoss().to(device)     
    pixel_criterion       = nn.MSELoss().to(device)     
     

    # initialize the weights for training 
    content_weight        = 1.0
    pixel_weight          = 0.01
    ## GAN configuration for weights
    adversarial_weight    = 0.001
    ## optimizer initialization
    g_optimizer           = optim.Adam(generator.parameters(),     0.0001, (0.9, 0.999))
    d_optimizer           = optim.Adam(discriminator.parameters(), 0.0001, (0.9, 0.999))
    p_optimizer           = optim.Adam(generator.parameters(),     0.0001, (0.9, 0.999)) 
      
      

    ## scheduler for gen and disc ----
    g_scheduler           = StepLR(g_optimizer, epochs // 2, 0.1) 
    d_scheduler           = StepLR(d_optimizer, epochs // 2, 0.1)  
    ## logger in place - configure as per requirements and space
    writer                = SummaryWriter(os.path.join("samples",  "logs", exp_name))

  
    exp_dir1 = os.path.join("samples", exp_name)
    exp_dir2 = os.path.join("results", exp_name)
## run for validation
if mode == "valid":
    
    exp_dir    = os.path.join("results", "test", exp_name)

    # path to model
    # need to change the name and the path as per the model saved path and directories
    model      = Generator().to(device)
    model_path = f"results/{exp_name}/g-best.pth"

    sr_dir     = f"results/test/{exp_name}"
    lr_dir     = f"data/Set5/LRbicx4"
    hr_dir     = f"data/Set5/GTmod12"
