from PIL import Image
from random import random, choice
import numpy as np
from io import BytesIO 
from scipy.ndimage.filters import gaussian_filter

class DataAugmenter:
    def __init__(self):
        self.jpeg_dict = {'pil': self.pil_jpg}

    def sample_continuous(self, s):
        if len(s) == 1:
            return s[0]
        if len(s) == 2:
            rg = s[1] - s[0]
            return random() * rg + s[0]
        raise ValueError("Length of iterable s should be 1 or 2.")

    def gaussian_blur(self, img, sigma):
        gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
        gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
        gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    def pil_jpg(self, img, compress_val):
        out = BytesIO()
        img = Image.fromarray(img)
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        out.close()
        return img

    def sample_discrete(self, s):
        if len(s) == 1:
            return s[0]
        return choice(s)

    def jpeg_from_key(self, img, compress_val, key):
        method = self.jpeg_dict[key]
        return method(img, compress_val)

    def data_augment(self, img, augment_chance):
        img = np.array(img)

        if random() < augment_chance:
            sig = self.sample_continuous([0.0, 2.0])
            self.gaussian_blur(img, sig)

        if random() < augment_chance:
            method =self. sample_discrete(['pil'])
            qual = self.sample_discrete([60, 70, 80, 90, 100])
            img = self.jpeg_from_key(img, qual, method)

        return Image.fromarray(img)