import copy
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision

from PIL import Image

import character_utility
from character import Char, Radical
from utility import pathstr


class RSDataset(Dataset):
    def __init__(self, charname2radicaljson, ignore_kana=False):
        self.charname2radicaljson = charname2radicaljson
        self.ignore_kana = ignore_kana
        
        self.items = []
        self.all_charnames = set()
        self.all_writernames = set()
        
        self.info = {
            "ignore_kana": ignore_kana,
            "datasets": [],
        }
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
    
    def add_item(self, image_path, charname, writername):
        if self.ignore_kana and (charname in character_utility.all_kanas):
            return
        
        char = Char.from_radicaljson(self.charname2radicaljson[charname])
        
        self.items.append((image_path, char, writername))
        self.all_charnames.add(charname)
        self.all_writernames.add(writername)
    
    def add_etlcdb(self, etlcdb_path, etlcdb_process_type, etlcdb_name):
        self.info["datasets"].append({
            "etlcdb_process_type": etlcdb_process_type,
            "etlcdb_name": etlcdb_name,
        })
        
        with open(pathstr(etlcdb_path, f"{etlcdb_name}.json")) as f:
            json_data = json.load(f)

        for item in json_data:
            relative_image_path = item["Path"] # ex) ETL4/5001/0x3042.png
            image_path = pathstr(etlcdb_path, etlcdb_process_type, relative_image_path)

            charname = item["Character"] # ex) "あ"

            serial_sheet_number = int(item["Serial Sheet Number"]) # ex) 5001
            writername = f"{etlcdb_name}_{serial_sheet_number}"

            self.add_item(image_path, charname, writername)

        
def create_dataloader(
    dataset,
    batch_size,
    shuffle_dataset,
    num_workers,
    shuffle_radicals_of_char,
    radicalname2idx,
    writername2idx,
):
    assert isinstance(dataset, RSDataset)
    
    def collate_fn(batch):
        nonlocal shuffle_radicals_of_char, radicalname2idx, writername2idx
        
        images = [None for _ in range(len(batch))]
        chars = [None for _ in range(len(batch))]
        
        if writername2idx is None:
            writerindices = None
        else:
            writerindices = [None for _ in range(len(batch))]
        
        for i, (image_path, char, writername) in enumerate(batch):
            images[i] = torchvision.transforms.functional.to_tensor(
                Image.open(image_path).convert("RGB")
            )
            
            chars[i] = copy.deepcopy(char)
            chars[i].register_radicalidx(radicalname2idx)
            if shuffle_radicals_of_char:
                random.shuffle(chars[i].radicals)
            
            if writerindices is not None:
                writerindices[i] = writername2idx[writername]
        
        images = torch.stack(images, 0)
        
        return images, chars, writerindices
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return dataloader
