import torch
from torch import nn
from torchvision import models
import torch.fft
class MultiStreamDenseNet(nn.Module):
    """ 
    This class represents a Two-stream densenet architecture with seperate architecture heads for the branches and the joint model
    """

    def __init__(self, pretrained=True):

        """ Init function
        Parameters
        ----------
        pretrained: bool
            If true will use weights from ImageNet else retrain from scratch
        """
        super(MultiStreamDenseNet, self).__init__()

        # Initialize the densenet blocks and additional layers
        dense_rgb = models.densenet161(pretrained=pretrained)
        dense_f = models.densenet161(pretrained=pretrained)

        features_rgb = list(dense_rgb.features.children())
        features_f= list(dense_f.features.children())

        self.enc_rgb = nn.Sequential(*features_rgb[0:8])
        self.enc_f = nn.Sequential(*features_f[0:8])

        self.linear=nn.Linear(384 + 384,1)
        self.linear_rgb=nn.Linear(384,1)
        self.linear_f = nn.Linear(384, 1)

        self.gavg_pool=nn.AdaptiveAvgPool2d(1)


    def forward(self, rgb_img, fourier_img):
        """ Propagate data through the network architecture. Expects 224x224x3 input RGB images and DFT input as tensors

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 

        Returns
        -------
        op, op_rgb, op_f: :py:class:`torch.Tensor`
        """

        x_rgb = rgb_img[:, [0,1,2], :, :]
        x_fourier = fourier_img[:, [0,1,2], :, :]
        
        enc_rgb = self.enc_rgb(x_rgb) # enc rgb
        enc_f = self.enc_f(x_fourier) # enc dft

        gap_rgb = self.gavg_pool(enc_rgb).squeeze() # pool rgb
        gap_f = self.gavg_pool(enc_f).squeeze() # pool dft 

        gap_rgb=gap_rgb.view(-1,384)
        gap_f=gap_f.view(-1,384) 

        gap_rgb = nn.Sigmoid()(gap_rgb)  # sigmoid rgb
        gap_f = nn.Sigmoid()(gap_f) # sigmoid dft 

        op_rgb=self.linear_rgb(gap_rgb) 
        op_f=self.linear_f(gap_f)

        op_rgb = nn.Sigmoid()(op_rgb) # head rgb
        op_f = nn.Sigmoid()(op_f) # head dft

        gap=torch.cat([gap_rgb,gap_f], dim=1)

        op = self.linear(gap)
        op = nn.Sigmoid()(op) # joint head
 
        return op, op_rgb, op_f
