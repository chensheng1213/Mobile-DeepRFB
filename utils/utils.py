import numpy as np
from PIL import Image
import torch.nn as nn

# Convert the image to RGB image
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

# resize the input image
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size
    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)
    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image, nw, nh

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0 # Image normalization processing
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'mobilenetv3': https://github.com/d-li14/mobilenetv3.pytorch.git
    }
    url = download_urls[backbone]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
