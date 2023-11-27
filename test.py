import os
import csv
import torch
from torchvision import transforms
from dataloader.synth_dataset_loader import CustomSynthImageLoader
from helpers.util import rgb_to_fft_tensor_filter_with_percentile 
import os
from torch.autograd import Variable
import warnings
import os
from options.test_options import TestOptions
warnings.filterwarnings("ignore")
from architectures.MultiStreamDenseNet import MultiStreamDenseNet
from architectures.OneStreamDenseNet import OneStreamDenseNet


architectures = dict({'two_stream': MultiStreamDenseNet, 'one_stream': OneStreamDenseNet}) # supported architectures
loss_types =  {"bce", "cmfl"}
allowed_spectras =  {"rgb", "fft", 'multi'}

def evaluate_model(test_args):

    SPECTRA = test_args.spectra
    DATA_DIR = test_args.dataroot
    PROTOCOL_DIR = test_args.protocol_dir
    PROTOCOL_NAME = test_args.protocol_name
    SELECTED_ARCHITECTURE = test_args.architecture
    IS_ALIGNED = test_args.is_aligned
    MODEL_PATH = test_args.model_path

    if SELECTED_ARCHITECTURE not in architectures.keys():
        raise Exception("invalid architecture selected. Please check your configuration")
    
    print(f"using data_dir: '{DATA_DIR}'")

    tensor_norms = dict()
    tensor_norms['rgb_norm'] = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    rz_func =  transforms.Resize(256)
    centercrop_func = transforms.CenterCrop(224)

    if not IS_ALIGNED:
        _img_transform =  transforms.Compose([
                rz_func,
                centercrop_func
            ])
    else:
        _img_transform =  transforms.Compose([
                transforms.Lambda(lambda img: img), # no need to do anything
            ])


    model_file = MODEL_PATH
    img_files = CustomSynthImageLoader(DATA_DIR, 
                                os.path.join(PROTOCOL_DIR, PROTOCOL_NAME, test_args.eval_split),
                                _img_transform,
                                custom_transform=None,
                                is_train=False
                                ).img_data
    
    network = architectures[SELECTED_ARCHITECTURE](pretrained=False)
    state_dict = torch.load(model_file, map_location='cpu')
    network.load_state_dict(state_dict['state_dict'])
    network.cuda()
    network.eval()

    results = []
    i = 0
    with torch.no_grad(): # to speed up computations you can implement support for batch here
        while i < len(img_files):
            img_file = img_files[i]
            input_image = img_file.load()
            input_image = _img_transform(input_image)
            rgb_tensor = tensor_norms['rgb_norm'](input_image).unsqueeze(0)
            fft_tensor = rgb_to_fft_tensor_filter_with_percentile(input_image).unsqueeze(0)

            if SPECTRA == "fft":
                outputs =  network(Variable(fft_tensor.cuda()))
            elif SPECTRA == "rgb":
                outputs =  network(Variable(rgb_tensor.cuda()))
            elif SPECTRA == "multi":
                outputs = network(Variable(rgb_tensor.cuda()), Variable(fft_tensor.cuda()))
            else:
                raise Exception("invalid config")
            
            if SPECTRA == "multi":
                outputs = outputs[0].cpu()
            else:
                outputs = outputs.cpu()
            if len(outputs) != 1:
                raise Exception("error")
            output = outputs[0]
            ci = img_file
            score = output.data.numpy().mean()
            result = [ci.id, ci.label, ci.fullpath, score]
            results.append(result)
            i += 1
    return results


if __name__ == "__main__":
    test_args = TestOptions().parse()
    output_dir = test_args.output_dir 
    results = evaluate_model(test_args)
    output_file = os.path.join(output_dir, test_args.result_file)
    with open(output_file, 'w+') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(results)