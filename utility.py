import os

import torch
import torchvision


def char2code(char):
    return format(ord(char), '#06x')

# def code2char(code):
#     chr(int(code, base=16))

def pathstr(t):
    return os.path.expanduser(t)

def save_images(images, path):
    m = torch.mean(images, dtype=torch.float).item()
    pad_value = 1 if m < 0.5 else 0
    
    grid = torchvision.utils.make_grid(images, pad_value=pad_value)
    image = torchvision.transforms.ToPILImage()(grid)
    image.save(path)
    return image

def save_single_image(image, path):
    image = torchvision.transforms.ToPILImage()(image)
    image.save(path)
    return image
