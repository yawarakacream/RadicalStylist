import copy
import json
import random
from typing import Any, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

import character_utility as charutil
from kanjivg import KvgContainer, KvgImageParameter
from radical import Radical
from utility import pathstr, read_image_as_tensor


class RSDataset(Dataset):
    kvgcontainer: KvgContainer
    ignore_kana: bool
    radical_depth: str
    writer_mode: str

    items: list[tuple[str, list[Radical], Optional[str]]] # (image_path, radicallist, writername)

    radicalname2idx: dict[str, int]
    writername2idx: Optional[dict[str, int]]

    info: dict[str, Any]

    def __init__(
        self,

        kvgcontainer: KvgContainer,
        radical_depth,
        ignore_kana,
        writer_mode,
    ):
        self.kvgcontainer = kvgcontainer
        self.ignore_kana = ignore_kana
        self.radical_depth = radical_depth
        self.writer_mode = writer_mode

        self.items = []

        self.radicalname2idx = {}
        self.writername2idx = None if writer_mode == "none" else {}

        self.info = {
            "ignore_kana": ignore_kana,
            "radical_depth": radical_depth,
            "datasets": [],
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, list[Radical], Optional[str]]:
        return self.items[idx]
    
    def add_item(self, image_path: str, char: Union[Radical, list[Radical]], writername: Optional[str]) -> None:
        if isinstance(char, Radical):
            radical = char

            if radical.name is None:
                raise Exception("cannot add nameless radical")

            if self.ignore_kana and (radical.name in charutil.all_kanas):
                return

            self.add_item(image_path, radical.get_radicals(self.radical_depth), writername)
            return

        radicallist: list[Radical] = char

        for radical in radicallist:
            self.radicalname2idx.setdefault(radical.name, len(self.radicalname2idx))

        if self.writer_mode == "none":
            assert writername is None
        else:
            assert writername is not None
            assert self.writername2idx is not None
            self.writername2idx.setdefault(writername, len(self.writername2idx))

        self.items.append((image_path, radicallist, writername))

    def add_kvg(self, mode, image_size, padding, stroke_width):
        self.info["datasets"].append({
            "dataset": "KVG",
            "mode": mode,
            "padding": padding,
            "stroke_width": stroke_width,
        })
        
        if self.writer_mode == "none":
            writername = None
        else:
            writername = f"KVG_{image_size}x,pad={padding},sw={stroke_width}"

        for charname in charutil.kanjis.all():
            if charname in charutil.all_kanas:
                continue

            kvg = self.kvgcontainer.get_kvg(charname)
            kvg_image_parameter = KvgImageParameter(image_size=image_size, padding=padding, stroke_width=stroke_width)

            stack = [kvg]
            while len(stack):
                kvg = stack.pop()

                if mode == "char":
                    pass
                elif mode == "radical":
                    stack += kvg.children
                else:
                    raise Exception(f"unknown mode: {mode}")

                if kvg.name is None:
                    continue
                
                image_path = kvg.get_image_path(kvg_image_parameter)
                radical = Radical.from_kvg(kvg, kvg_image_parameter)
                self.add_item(image_path, radical, writername)

    def add_etlcdb(self, etlcdb_path: str, etlcdb_process_type: str, etlcdb_name: str, radical_position: KvgImageParameter) -> None:
        self.info["datasets"].append({
            "dataset": "etlcdb",
            "etlcdb_process_type": etlcdb_process_type,
            "etlcdb_name": etlcdb_name,
        })

        with open(pathstr(etlcdb_path, f"{etlcdb_name}.json")) as f:
            json_data = json.load(f)
        
        for item in json_data:
            relative_image_path = item["Path"] # ex) ETL4/5001/0x3042.png
            image_path = pathstr(etlcdb_path, etlcdb_process_type, relative_image_path)

            charname = item["Character"] # ex) "あ"

            if self.writer_mode == "none":
                writername = None

            elif self.writer_mode == "dataset":
                writername = etlcdb_name

            elif self.writer_mode == "all":
                serial_sheet_number = int(item["Serial Sheet Number"]) # ex) 5001
                writername = f"{etlcdb_name}_{serial_sheet_number}"

            else:
                raise Exception(f"unknown writer_mode: {self.writer_mode}")

            if self.ignore_kana and (charname in charutil.all_kanas):
                continue

            kvg = self.kvgcontainer.get_kvg(charname)
            radical = Radical.from_kvg(kvg, radical_position)

            self.add_item(image_path, radical, writername)

    def create_dataloader(
        self,
        batch_size,
        shuffle_dataset,
        num_workers,
        shuffle_radicallist_of_char,
    )-> DataLoader[tuple[torch.Tensor, list[Radical], Optional[list[int]]]]:
        def collate_fn(batch: list[tuple[str, list[Radical], str]]) -> tuple[torch.Tensor, list[Radical], Optional[list[int]]]:
            nonlocal shuffle_radicallist_of_char

            images: Any = [None for _ in range(len(batch))]
            radicallists: list = [None for _ in range(len(batch))]
            writerindices: Optional[list] = None
            if self.writername2idx is not None:
                writerindices = [None for _ in range(len(batch))]

            for i, (image_path, radicallist, writername) in enumerate(batch):
                images[i] = read_image_as_tensor(image_path)

                radicallists[i] = copy.deepcopy(radicallist)
                for r in radicallists[i]:
                    r.set_idx(self.radicalname2idx)
                if shuffle_radicallist_of_char:
                    random.shuffle(radicallists[i])

                if writerindices is not None:
                    assert self.writername2idx is not None
                    writerindices[i] = self.writername2idx[writername]

            images = torch.stack(images, 0)

            return images, radicallists, writerindices
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle_dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
