import json
import os

import torch

import torchvision
from torchvision.transforms import functional as TVF

from PIL import Image


def pathstr(*s: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.join(*s)))


def save_images(images, path):
    m = torch.mean(images, dtype=torch.float).item()
    pad_value = 1 if m < 0.5 else 0
    
    grid = torchvision.utils.make_grid(images, pad_value=pad_value)
    image = torchvision.transforms.ToPILImage()(grid)
    image.save(path)
    return image


def save_single_image(image, path):
    image = TVF.to_pil_image(image)
    image.save(path)
    return image


def read_image_as_tensor(image_path):
    return TVF.to_tensor(Image.open(image_path).convert("RGB"))


def rgb_to_grayscale(image):
    return TVF.rgb_to_grayscale(image, num_output_channels=3)


def char2code(char: str) -> str:
    return format(ord(char), "#06x")


# def code2char(code: str) -> str:
#     return chr(int(code, base=16))


def create_charname2radicaljson(radicals_data_path: str):
    with open(radicals_data_path) as f:
        radicals_data = json.load(f)
    return {radical["name"]: radical for radical in radicals_data}
