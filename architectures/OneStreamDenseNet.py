import torch
from torch import nn
from torchvision import models
import numpy as np
import torch.fft
from torchvision import transforms
class OneStreamDenseNet(nn.Module):
    """ 
    This class represents a One-stream densenet architecture with seperate architecture heads for the branches and the joint model
    """

    def __init__(self, pretrained=True):

        """ Init function
        Parameters
        ----------
        img: :py:class:`torch.Tensor` 

        Returns
        -------
        op, op_rgb, op_f: :py:class:`torch.Tensor`
        """
        super(OneStreamDenseNet, self).__init__()

        dense_rgb = models.densenet161(pretrained=pretrained)
        features_rgb = list(dense_rgb.features.children())
        self.enc_rgb = nn.Sequential(*features_rgb[0:8])
        self.linear_rgb=nn.Linear(384,1)
        self.gavg_pool=nn.AdaptiveAvgPool2d(1)


    def forward(self, img):
        """ Propagate data through the network architecture. Expects 224x224x3 input RGB images and DFT input as tensors

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 

        Returns
        -------
        op, op_rgb, op_f: :py:class:`torch.Tensor`
        """

        x_rgb = img[:, [0,1,2], :, :]  # could also be FFTN image
        enc_rgb = self.enc_rgb(x_rgb)
        gap_rgb = self.gavg_pool(enc_rgb).squeeze() 
        gap_rgb=gap_rgb.view(-1,384)
        gap_rgb = nn.Sigmoid()(gap_rgb) 
        op_rgb=self.linear_rgb(gap_rgb)
        op_rgb = nn.Sigmoid()(op_rgb)
        return op_rgb