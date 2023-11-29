import os
import sys
from typing import NamedTuple, Sequence

import numpy as np

import torch

import torchvision
from torchvision.transforms import functional as TVF

from PIL import Image, ImageMath


def pathstr(*s: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.join(*s)))


def add_sys_path(path: str) -> None:
    path = pathstr(path)
    if not any(map(lambda p: p == path, sys.path)):
        sys.path.append(path)


def char2code(char: str) -> str:
    return format(ord(char), "#06x")


# def code2char(code: str) -> str:
#     return chr(int(code, base=16))


def save_images(images, path: str):
    m = torch.mean(images, dtype=torch.float).item()
    pad_value = 1 if m < 0.5 else 0
    
    grid = torchvision.utils.make_grid(images, pad_value=pad_value)
    image = torchvision.transforms.ToPILImage()(grid)
    image.save(path)
    return image


def save_single_image(image, path: str):
    image = TVF.to_pil_image(image)
    image.save(path)
    return image


def read_image_as_tensor(image_path: str) -> torch.Tensor:
    return TVF.to_tensor(Image.open(image_path).convert("RGB"))


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    return TVF.rgb_to_grayscale(image, num_output_channels=3)


class LImageCompositionResult(NamedTuple):
    image: Image.Image # 合成結果
    n_blended: int # 被ったピクセルの数


# I から L への変換がうまく動かない
# https://github.com/python-pillow/Pillow/issues/3011
def convert_to_L(image: Image.Image):
    if image.mode == "L":
        pass
    elif image.mode == "RGB" or image.mode == "RGBA":
        image = image.convert("L")
    elif image.mode == "I":
        image = ImageMath.eval("image >> 8", image=image).convert("L")
        assert isinstance(image, Image.Image)
    else:
        raise Exception(f"unknown mode: {image.mode}")
    assert image.mode == "L"
    return image


def compose_L_images(images: Sequence[Image.Image]) -> LImageCompositionResult:
    if len(images) == 0:
        raise Exception()
    
    image_size = images[0].size
    
    image = Image.new("L", image_size)
    sum_of_images = np.zeros(image_size, np.int32)
    for i in range(len(images)):
        if images[i].mode != "L":
            raise Exception(f"images[{i}] is not L")
        if images[i].size != image_size:
            raise Exception(f"invalid image size: {images[i].size} at {i}, {image_size} at 0")
        
        image = ImageMath.eval("image | image_i", image=image, image_i=images[i]).convert("L")
        assert isinstance(image, Image.Image)
        sum_of_images += np.array(images[i]) > 0 # 適当

    n_blended = np.count_nonzero(sum_of_images > 1)

    return LImageCompositionResult(image=image, n_blended=n_blended)
