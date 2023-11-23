from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from character_decomposer import BoundingBoxDecomposer, BoundingBoxRadical
from kanjivg import KvgContainer
from utility import pathstr, read_image_as_tensor


class KvgDatasetRecord:
    charnames: set[str]
    kvgcontainer: KvgContainer

    mode: Union[Literal["character"], Literal["radical"]]
    image_size: int
    padding: int
    stroke_width: int

    def __init__(
        self,

        charnames: Iterable[str],
        kvg_path: str,

        mode: Union[Literal["character"], Literal["radical"]],
        image_size: int,
        padding: int,
        stroke_width: int,
    ):
        self.charnames = set(charnames)
        self.kvgcontainer = KvgContainer(kvg_path)
        self.mode = mode
        self.image_size = image_size
        self.padding = padding
        self.stroke_width = stroke_width

    @property
    def info(self) -> dict:
        return {
            "name": "kvg",
            "mode": self.mode,
            "image_size": self.image_size,
            "padding": self.padding,
            "stroke_width": self.stroke_width,
        }
    

class EtlcdbDatasetRecord:
    etlcdb_path: str
    etlcdb_process_type: str
    etlcdb_name: str
    charnames: set[str]

    def __init__(self, etlcdb_path: str, etlcdb_process_type: str, etlcdb_name: str, charnames: Iterable[str]):
        self.etlcdb_path = etlcdb_path
        self.etlcdb_process_type = etlcdb_process_type
        self.etlcdb_name = etlcdb_name
        self.charnames = set(charnames)

    @property
    def info(self) -> dict:
        return {
            "name": "etlcdb",
            "etlcdb_process_type": self.etlcdb_process_type,
            "etlcdb_name": self.etlcdb_name,
        }


@dataclass(frozen=True)
class DatasetItem:
    charname: str
    writername: Optional[str]
    image_path: str
    radicallist: list[BoundingBoxRadical]


class RSDataset(Dataset):
    decomposer: CharacterDecomposer
    writer_mode: WriterMode
    image_size: int

    items: list[DatasetItem]

    radicalname2idx: dict[str, int]
    writername2idx: dict[str, int]

    info: dict

    def __init__(
        self,

        decomposer: CharacterDecomposer,
        writer_mode: WriterMode,
        image_size: int,
    ):
        self.decomposer = decomposer
        self.writer_mode = writer_mode
        self.image_size = image_size

        self.items = []

        self.radicalname2idx = {}
        self.writername2idx = {}

        self.info = {
            "decomposer": decomposer.info,
            "writer_mode": writer_mode,
            "image_size": image_size,
            "data": [],
        }
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self.items[idx]
    
    def add_item(self, item: DatasetItem):
        self.items.append(item)

        for radical in item.radicallist:
            self.radicalname2idx.setdefault(radical.name, len(self.radicalname2idx))
        
        if self.writer_mode != "none":
            assert item.writername is not None
            self.writername2idx.setdefault(item.writername, len(self.writername2idx))

        image = Image.open(item.image_path)
        if image.size[0] != self.image_size or image.size[1] != self.image_size:
            raise Exception(f"invalid image size: {image.size} at {item.image_path}")

    def add_items(self, items: list[DatasetItem]):
        for item in items:
            self.add_item(item)

    def add_kvg(self, record: KvgDatasetRecord):
        self.info["data"].append(record.info)

        if self.writer_mode == "none":
            writername = None
        else:
            writername = f"KVG_{self.image_size}x,pad={record.padding},sw={record.stroke_width}"

        for charname in record.charnames:
            kvg = record.kvgcontainer.get_kvg(charname)

            stack = [kvg]
            while len(stack):
                kvg = stack.pop()

                if record.mode == "character":
                    pass
                elif record.mode == "radical":
                    stack += kvg.children
                else:
                    raise Exception(f"unknown mode: {record.mode}")

                if not self.decomposer.is_kvgid_registered(kvg.kvgid):
                    continue

                assert kvg.name is not None
                
                image_path = kvg.get_image_path(image_size=self.image_size, padding=record.padding, stroke_width=record.stroke_width)

                radicallist = self.decomposer.get_decomposition_by_kvgid(kvg.kvgid)

                self.add_item(DatasetItem(charname=kvg.name, writername=writername, image_path=image_path, radicallist=radicallist))

    def add_etlcdb(self, record: EtlcdbDatasetRecord):
        self.info["data"].append(record.info)

        with open(pathstr(record.etlcdb_path, f"{record.etlcdb_name}.json")) as f:
            json_data = json.load(f)

        items: list[DatasetItem] = []

        for item in json_data:
            relative_image_path = item["Path"] # ex) ETL4/5001/0x3042.png
            image_path = pathstr(record.etlcdb_path, record.etlcdb_process_type, relative_image_path)

            charname = item["Character"] # ex) "あ"
            assert isinstance(charname, str)

            if charname not in record.charnames:
                continue

            if self.writer_mode == "none":
                writername = None
            elif self.writer_mode == "dataset":
                writername = record.etlcdb_name
            elif self.writer_mode == "all":
                serial_sheet_number = int(item["Serial Sheet Number"]) # ex) 5001
                writername = f"{record.etlcdb_name}_{serial_sheet_number}"
            else:
                raise Exception(f"unknown writer_mode: {self.writer_mode}")

            radicallist = self.decomposer.get_decomposition_by_charname(charname)

            self.add_item(DatasetItem(charname=charname, writername=writername, image_path=image_path, radicallist=radicallist))

        return items
    
    def add_by_record(self, record: Union[KvgDatasetRecord, EtlcdbDatasetRecord]):
        if isinstance(record, KvgDatasetRecord):
            self.add_kvg(record)
        elif isinstance(record, EtlcdbDatasetRecord):
            self.add_etlcdb(record)
        else:
            raise Exception(f"unknown record: {record}")
    
    def create_dataloader(
        self,
        batch_size,
        shuffle_dataset,
        num_workers,
        shuffle_radicallist_of_char,
    )-> DataLoader[tuple[torch.Tensor, list[Radical], Optional[list[int]]]]:
        def collate_fn(batch: list[DatasetItem]) -> tuple[torch.Tensor, list[Radical], Optional[list[int]]]:
            nonlocal shuffle_radicallist_of_char

            images: Any = [None for _ in range(len(batch))]
            radicallists: list = [None for _ in range(len(batch))]
            writerindices: Optional[list] = None
            if self.writername2idx is not None:
                writerindices = [None for _ in range(len(batch))]

            for i, item in enumerate(batch):
                images[i] = read_image_as_tensor(item.image_path)

                radicallists[i] = copy.deepcopy(item.radicallist)
                for r in radicallists[i]:
                    r.set_idx(self.radicalname2idx)
                if shuffle_radicallist_of_char:
                    random.shuffle(radicallists[i])

                if writerindices is None:
                    assert self.writername2idx is None
                    assert item.writername is None
                else:
                    assert self.writername2idx is not None
                    assert item.writername is not None
                    writerindices[i] = self.writername2idx[item.writername]

            images = torch.stack(images, 0)

            return images, radicallists, writerindices
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle_dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


DatasetRecord = Union[KvgDatasetRecord, EtlcdbDatasetRecord]
CharacterDecomposer = BoundingBoxDecomposer
WriterMode = Union[Literal["none"], Literal["dataset"], Literal["all"]]
Radical = BoundingBoxRadical
