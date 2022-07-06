import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

__all__ = [
     "Generator", "Discriminator"
    "ContentLoss" , "ResidualConvBlock"
]

 
class Generator(nn.Module):
    ## defining the generator model
    ## extending the class
    ## initializing sequential - network expecting 64x3

    def __init__(self) -> None:
        super(Generator, self).__init__()

        self.convolutional_block1 = nn.Sequential(
            nn.Conv2d(3, 64, (9, 9), (1, 1), (4, 4)),
            nn.PReLU()
        )

        ## adding resnet conv block of 16

        res_trunk = []
        for _ in range(16):
            res_trunk.append(ResidualConvBlock(64))
        self.res_trunk = nn.Sequential(*res_trunk)


        self.convolutional_block2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64)
        )


        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU()

        )


        self.convolutional_block3 = nn.Conv2d(64, 3, (9, 9), (1, 1), (4, 4))


        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


    def _forward_impl(self, x: Tensor) -> Tensor:
        ## defining forward pass -> 3 convolutional blocks
        out1 = self.convolutional_block1(x)
        out = self.res_trunk(out1)
        out2 = self.convolutional_block2(out)
        output = out1 + out2

        output = self.upsampling(output)
        output = self.convolutional_block3(output)

        return output

    def _initialize_weights(self) -> None:
        ## initialize the weights 
        ## adding provision for batch normalization 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1



class Discriminator(nn.Module):
    ## defining discriminator
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        ## define forward pass
        output = self.features(x)
        output = torch.flatten(output, 1)
        output = self.classifier(output)

        return output

class ContentLoss(nn.Module):

    ## defining content loss class
    ## feature extractors - till 36
    def __init__(self) -> None:
        super(ContentLoss, self).__init__()

        vgg19_model = models.vgg19(pretrained=True, num_classes=1000).eval()

        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:36])

        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        hr = (hr - self.mean) / self.std
        sr = (sr - self.mean) / self.std
        

        mse_loss = F.mse_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return mse_loss

class ResidualConvBlock(nn.Module):
    ## get residual block

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rc_block = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        output = self.rc_block(x)
        output = output + identity

        return output


