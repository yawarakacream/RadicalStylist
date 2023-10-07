import copy
import json
import os
from abc import abstractmethod
from dataclasses import dataclass, asdict as dc_asdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision

from PIL import Image

import character
from utility import pathstr


@dataclass
class Radical:
    name: str
    idx: Optional[int]
    center_x: float
    center_y: float
    width: float
    height: float
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.idx = kwargs.get("idx", None)
        
        if "center_x" in kwargs:
            self.center_x = kwargs["center_x"]
            self.center_y = kwargs["center_y"]
            self.width = kwargs["width"]
            self.height = kwargs["height"]
            
        elif "left" in kwargs:
            left = kwargs["left"]
            right = kwargs["right"]
            top = kwargs["top"]
            bottom = kwargs["bottom"]
            
            self.center_x = (left + right) / 2
            self.center_y = (top + bottom) / 2
            self.width = right - left
            self.height = bottom - top
        
        else:
            raise Exception(f"invalid arguments: {kwargs}")
    
    def from_radicaljson(dct):
        name = dct["name"]
        if dct["part"] is not None:
            name += f'_{dct["part"]}'
        
        bounding = dct["bounding"]

        left = bounding["left"]
        right = bounding["right"]
        top = bounding["top"]
        bottom = bounding["bottom"]

        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        width = right - left
        height = bottom - top
        
        return Radical(name, center_x=center_x, center_y=center_y, width=width, height=height)
    
    def from_dict(dct):
        return Radical(**dct)
    
    def to_dict(self):
        return dc_asdict(self)

    @property
    def left(self):
        return self.center_x - self.width / 2
    
    @property
    def right(self):
        return self.center_x + self.width / 2
    
    @property
    def top(self):
        return self.center_y - self.height / 2
    
    @property
    def bottom(self):
        return self.center_y + self.height / 2


@dataclass
class Char:
    name: Optional[str]
    radicals: list[Radical]
    
    def from_radicaljson(dct):
        def get_radicals(dct):
            from_children = []
            for d in dct["children"]:
                c = get_radicals(d)
                if c is None:
                    from_children = None
                    break
                from_children += c
            
            from_name = dct["name"] and [Radical.from_radicaljson(dct)]
            
            if (from_children is not None) and len(from_children):
                # 例) 三 = 三_1 + 三_2 = 一 + 三_2 のとき 三_1 + 三_2 を採用する
                if (len(from_children) == 1) and (from_name is not None):
                    return from_name
                
                return from_children
            
            return from_name
        
        name = dct["name"]
        radicals = get_radicals(dct)
        return Char(name, radicals)
    
    def from_dict(dct):
        name = dct["name"]
        radicals = [Radical.from_dict(r) for r in dct["radicals"]]
        return Char(name, radicals)
    
    def to_dict(self):
        name = self.name
        radicals = [r.to_dict() for r in self.radicals]
        return {"name": name, "radicals": radicals}
    
    def to_formula_string(self):
        return f"{self.name} = {' + '.join(map(lambda r: r.name, self.radicals))}"


def create_data_loader(batch_size, shuffle, num_workers, datasets, chars_filter, charname2radicaljson):
    pil_to_tensor = torchvision.transforms.ToTensor()
    
    def collate_fn(batch):
        nonlocal pil_to_tensor
        
        images = [None for _ in range(len(batch))]
        chars = [None for _ in range(len(batch))]
        writers = [None for _ in range(len(batch))]
        
        for i, (image_path, char, writer) in enumerate(batch):
            images[i] = pil_to_tensor(Image.open(image_path).convert("RGB"))
            chars[i] = copy.deepcopy(char)
            writers[i] = copy.deepcopy(writer)
        
        images = torch.stack(images, 0)
        
        return images, chars, writers
    
    dataset = ConcatDataset(datasets, chars_filter, charname2radicaljson)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return data_loader, dataset.all_charnames, dataset.all_writers


class ConcatDataset(Dataset):
    def __init__(self, datasets, chars_filter, charname2radicaljson):
        self.datasets = datasets
        
        self.all_charnames = set()
        self.all_writers = set()
        self.items = []
        for dataset in datasets:
            for image_path, char, writer in dataset:
                if (chars_filter is not None) and (char not in chars_filter):
                    continue
                
                # かなは学習しない
                if char in character.all_kanas:
                    continue
                
                self.all_charnames.add(char)
                self.all_writers.add(writer)
                
                char = Char.from_radicaljson(charname2radicaljson[char])
                self.items.append((image_path, char, writer))
        
        if chars_filter is not None:
            uncontained_chars = [char for char in chars_filter if char not in self.all_charnames]
            if len(uncontained_chars):
                raise Exception(f"these chars are not supported: {uncontained_chars}")
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


class RSDataset(Dataset):
    rs_dataset_info: dict


class EtlcdbDataset(RSDataset):
    def __init__(self, etlcdb_path, etlcdb_process_type, etlcdb_names):
        self.rs_dataset_info = {
            "etlcdb_process_type": etlcdb_process_type,
            "etlcdb_names": etlcdb_names,
        }
        
        self.items = []
        
        for etlcdb_name in etlcdb_names:
            json_path = pathstr(etlcdb_path, f"{etlcdb_name}.json")
            with open(json_path) as f:
                json_data = json.load(f)

            for item in json_data:
                relative_image_path = item["Path"] # ex) ETL4/5001/0x3042.png
                image_path = pathstr(etlcdb_path, etlcdb_process_type, relative_image_path)
                
                charname = item["Character"] # ex) "あ"
                
                serial_sheet_number = int(item["Serial Sheet Number"]) # ex) 5001
                writer = f"{etlcdb_name}_{serial_sheet_number}"
                
                self.items.append((image_path, charname, writer))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
