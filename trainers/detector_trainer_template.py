from ast import Try
from helpers.data_augmentation import DataAugmenter
from dataloader.synth_dataset_loader import CustomSynthImageLoader
from loss.cmfl_loss import CMFL
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy
import warnings
import os
warnings.filterwarnings("ignore")
from architectures.MultiStreamDenseNet import MultiStreamDenseNet
from architectures.OneStreamDenseNet import OneStreamDenseNet

class TrainerConfig():
    """ 
    This class configures the training of the network and support the different architecture and experiments
    """

    def __init__(self, train_args):
        self.args = train_args
        
        architectures = dict({'two_stream': MultiStreamDenseNet, 'one_stream': OneStreamDenseNet}) # supported architectures
        loss_types =  {"bce", "cmfl"}
        allowed_spectras =  {"rgb", "fft", 'multi'}

        self.SPECTRA = train_args.spectra
        self.DATA_DIR = train_args.dataroot
        self.AUGMENT_CHANCE = train_args.augment_chance
        self.PROTOCOL_DIR = train_args.protocol_dir
        self.PROTOCOL_NAME = train_args.protocol_name
        self.SELECTED_ARCHITECTURE = train_args.architecture
        self.IS_ALIGNED = train_args.is_aligned
        self.SHOULD_AUGMENT_VALIDATION = train_args.augment_val
        self.LOSS_TYPE = train_args.loss
        self.PRETRAINED = train_args.pretrained
        self.DO_CROSS_VALIDATION = True

        #==============================================================================
        # Initialize the torch dataset, subselect channels from the pretrained files if needed.
        SELECTED_CHANNELS = [0,1,2] 
        ####################################################################

        if self.SELECTED_ARCHITECTURE not in architectures.keys():
            raise Exception("invalid architecture selected. Please check your configuration")

        if self.LOSS_TYPE not in loss_types:
            raise Exception("Unsupported loss type")

        if self.SPECTRA not in allowed_spectras:
            raise Exception("Unsupported spectra")

        phases = ['train','val']
        phase_files = {"train":train_args.train_split,"val":train_args.val_split}


        rz_func =  transforms.Resize(256)
        centercrop_func = transforms.CenterCrop(224)
        randomcrop_func = transforms.RandomResizedCrop(224)
        flip_func = transforms.RandomHorizontalFlip(p=0.5)

        _dataaugmentor = DataAugmenter()
        def data_augment(img):
            return _dataaugmentor.data_augment(img, augment_chance=self.AUGMENT_CHANCE)

        def data_augment_val(img):
            if self.SHOULD_AUGMENT_VALIDATION:
                return _dataaugmentor.data_augment(img, augment_chance=self.AUGMENT_CHANCE)
            else: 
                return img

        img_transform = {}

        if not self.IS_ALIGNED:
            img_transform['train'] = transforms.Compose([
                    #transforms.ToPILImage(),
                    rz_func,
                    transforms.Lambda(lambda img: data_augment(img)),
                    randomcrop_func,
                    flip_func])

            img_transform['val'] =  transforms.Compose([
                    #transforms.ToPILImage(),
                    rz_func,
                    transforms.Lambda(lambda img: data_augment_val(img)),
                    centercrop_func
                ])
            
        else:
            img_transform['train'] = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.Lambda(lambda img: data_augment(img)),
                flip_func])

            img_transform['val'] =  transforms.Compose([
                    #transforms.ToPILImage(),
                    transforms.Lambda(lambda img: data_augment_val(img))
                ])

        self.dataset={}

        for phase in phases:
            self.dataset[phase] = CustomSynthImageLoader(self.DATA_DIR, 
                                os.path.join(self.PROTOCOL_DIR, self.PROTOCOL_NAME, phase_files[phase]),
                                img_transform[phase],
                                custom_transform = custom_function,
                                is_train=True)

        # Load the architecture
        self.network = architectures[self.SELECTED_ARCHITECTURE](pretrained=self.PRETRAINED)

        for name,param in self.network.named_parameters():
            param.requires_grad = True

        # loss definitions

        self.criterion_bce= nn.BCELoss()
        self.criterion_cmfl = CMFL(alpha=1, gamma= 3, binary= False, multiplier=2)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()),lr = train_args.lr, weight_decay=train_args.weight_decay)

                                    
    def compute_loss(self, network,img, labels, device):
        """
        Compute the losses, given the network, data and labels and 
        device in which the computation will be performed. 
        """

        if self.SELECTED_ARCHITECTURE == "two_stream":
            if self.LOSS_TYPE == "bce":
                return self.two_stream_loss_bce(network, img, labels, device)
            elif self.LOSS_TYPE == "cmfl":
                return self.two_stream_loss_cmfl(network,img, labels, device)
            else:
                raise Exception("unsupported loss")
        elif self.SELECTED_ARCHITECTURE == "one_stream":
            return self.one_stream_loss(network, img, labels, device)
        else:
            Exception("Loss is not defined for the selected architecture")

    def two_stream_loss_cmfl(self, network,img, labels, device):
        """
        Loss for the two stream architecture using CMFL
        """

        beta = 0.5
        imagesv = Variable(img['image'].to(device))
        image_foruier = Variable(img['fourier'].to(device))

        labelsv_binary = Variable(labels['binary_target'].to(device))

        op, op_rgb, op_f = network(imagesv, image_foruier)

        loss_cmfl = self.criterion_cmfl(op_rgb,op_f,labelsv_binary.unsqueeze(1).float())
        loss_bce = self.criterion_bce(op,labelsv_binary.unsqueeze(1).float())
        loss= beta*loss_cmfl +(1-beta)*loss_bce

        return loss

    def two_stream_loss_bce(self, network,img, labels, device):
        """
        Loss for the two stream architecture using BCE
        """
        imagesv = Variable(img['image'].to(device))
        image_foruier = Variable(img['fourier'].to(device))

        labelsv_binary = Variable(labels['binary_target'].to(device))

        op, _, _ = network(imagesv, image_foruier)

        loss_bce = self.criterion_bce(op,labelsv_binary.unsqueeze(1).float())

        return loss_bce

    def one_stream_loss(self, network,img, labels, device):
        """
        Loss for the one stream architecture using BCE
        """
        labelsv_binary = Variable(labels['binary_target'].to(device))
            
        if self.SPECTRA == "rgb":
            imagesv = Variable(img['image'].to(device))
            op = network(imagesv)
        elif self.SPECTRA == "fft":
            image_foruier = Variable(img['fourier'].to(device))
            op = network(image_foruier)
        loss_bce = self.criterion_bce(op,labelsv_binary.unsqueeze(1).float())
        return loss_bce

# Targets: 1 = Real, 0 = attack
def custom_function(img_rgb_tensor, img_fft_tensor, target, img_name):
    img = {}
    img['image'] = img_rgb_tensor
    img['fourier'] = img_fft_tensor
    labels={}
    labels['binary_target']=target
    labels['img_name']=img_name
    
    return img, labels