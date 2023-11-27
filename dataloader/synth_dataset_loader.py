import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from copy import deepcopy
import os
from helpers.util import rgb_to_fft_tensor_filter_with_percentile
import csv
from sqlite3 import NotSupportedError
import os
from functools import reduce
import operator
from random import shuffle

class CustomSynthImageLoader(data.Dataset):
    """
    Custom dataloader which loads data from a csv file
    """
    def __init__(self, data_base_dir, csv_file_path, preprocess_transform, custom_transform, is_train,  purposes=['real', 'attack']):
        """ 
        Parameters
        ----------
        network: str
                Base directory to where data is located. Base dir + relative path in csv should give the sample paths
        csv_file_path: str
                Path to csv file to read sample data from
        preprocess_transform: Torch transformer
                Transformation to be applied as preprocessing to the RGB input image
        custom_transform: function
                custom wrapper function
        is_train: bool
        """
        self.data_base_dir = data_base_dir
        self.is_train = is_train
        self.preprocess_transform = preprocess_transform 
        self.custom_transform = custom_transform
        self.purposes = purposes

        self.img_data = self._objects(csv_file_path, self.purposes)

        self.tensor_norms = dict()

        self.tensor_norms['rgb_norm'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
    def _objects(self, csv_file_path, purposes):
        all_data = self._get_samples_from_csv(csv_file_path)
        data_lists = []
        for purpose in purposes:
            data = self.filter_for_purposes(all_data, purpose)
            data_lists.append(data)
        if len(purposes) != 2:
            raise NotSupportedError()
 
        if self.is_train: # ensure equal size if this is the training partition
            min_length = min(len(data_lists[0]), len(data_lists[1]))
            data_lists[0] = data_lists[0][:min_length]
            data_lists[1] = data_lists[1][:min_length]
        data_lists = reduce(operator.concat, data_lists)
        shuffle(data_lists)
        return data_lists
    
    def filter_for_purposes(self, data_samples, purpose):
        if purpose == "real":
            target_label = 1
        elif purpose == "attack":
            target_label =  0
        else:
            raise NotImplementedError("unsurposed purpose")
        result_list = [data for data in data_samples if data.label == target_label]
        return result_list
    
    def validate_purposes(self, purposes):
        if not isinstance(purposes, list):
            raise NotSupportedError()
        allowed = ['real', 'attack']
        for purpose in purposes:
            if purpose not in allowed:
                raise NotSupportedError()

    def _get_samples_from_csv(self, csv_file_path):
        samples = []
        with open(csv_file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    data_path = os.path.join(self.data_base_dir, row[1])
                    if not os.path.isfile(data_path):
                        continue # skipping if file does not exists
                    sample = SynthDataSample(data_path, row[0], row[2])
                    samples.append(sample)
        return samples
    
    def __len__(self):
        return len(self.img_data)


    def __getitem__(self, idx):
        datasample = self.img_data[idx] # datasample
        image = datasample.load()
        image = self.preprocess_transform(image)
        rgb_tensor = self.tensor_norms['rgb_norm'](image)
        fft_image = rgb_to_fft_tensor_filter_with_percentile(image)
        label = datasample.label
        return self.custom_transform(rgb_tensor, fft_image, label, datasample.id)
    
class SynthDataSample: 
    def __init__(self, fullpath, image_id, attack_type):
        self.id = image_id
        self.label  =  self.attack_type_to_label(attack_type)
        self.fullpath = fullpath

    def load(self):
        return Image.open(self.fullpath).convert('RGB')

    # Targets: 1 = Real, 0 = attack
    def attack_type_to_label(self, attack_type):
        if not attack_type:
            return 1 # real
        return 0 # attack