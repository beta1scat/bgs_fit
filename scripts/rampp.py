'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

if __name__ == "__main__":
    image_path = '/root/ros_ws/src/data/table.jpg'
    image_size = 384
    model_path = '/root/ros_ws/src/data/models/ram_plus_swin_large_14m.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=image_size)

    #######load model
    model = ram_plus(pretrained=model_path, image_size=image_size, vit='swin_l')
    model.eval()
    model = model.to(device)

    image = transform(Image.open(image_path)).unsqueeze(0).to(device)

    res = inference(image, model)
    print(res)
    print("Image Tags: ", res[0])
    print("图像标签: ", res[1])