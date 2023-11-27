import numpy as np
import torch.utils.data as data
from copy import deepcopy
import torch
import os

def rgb_to_fft_tensor_filter_with_percentile(image):
    im = deepcopy(np.asarray(image))
    # approach inspired from https://arxiv.org/pdf/1907.06515.pdf
    im = im.astype(np.float32)
    im = im/255.0
    for i in range(3):
        img_channel = im[:,:,i]
        fft_img = np.fft.fft2(img_channel)
        fft_img = np.log(np.abs(fft_img)+1e-3)
        fft_min = np.percentile(fft_img,5)
        fft_max = np.percentile(fft_img,95)
        fft_img = (fft_img - fft_min)/(fft_max - fft_min)
        fft_img = (fft_img-0.5)*2
        fft_img[fft_img<-1] = -1
        fft_img[fft_img>1] = 1
        fft_img = np.fft.fftshift(fft_img)
        im[:,:,i] = fft_img
    transposed_fft = np.transpose(im, (2,0,1))
    return torch.Tensor(transposed_fft)

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
