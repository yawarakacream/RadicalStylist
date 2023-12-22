from __future__ import annotations

import copy
import glob
import json
import os
import random
from typing import Final, Iterable, Literal, NamedTuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import functional as TVF

from PIL import Image

from character_decomposer import BoundingBoxDecomposer, ClusteringLabelDecomposer, IdentityDecomposer
from kanjivg import KvgContainer
from radical import Radical
from utility import pathstr, convert_to_L, compose_L_images


class KvgDatasetProvider:
    kvgcontainer: Final[KvgContainer]
    charnames: Final[set[str]]

    mode: Final[Union[Literal["character"], Literal["radical"]]]
    padding: Final[int]
    stroke_width: Final[int]

    def __init__(
        self,

        kvg_path: str,
        charnames: Iterable[str],

        mode: Union[Literal["character"], Literal["radical"]],
        padding: int,
        stroke_width: int,
    ):
        self.kvgcontainer = KvgContainer(kvg_path)
        self.charnames = set(charnames)

        self.mode = mode
        self.padding = padding
        self.stroke_width = stroke_width

    @property
    def info(self) -> dict:
        return {
            "mode": self.mode,
            "padding": self.padding,
            "stroke_width": self.stroke_width,
        }

    def append_to(self, dataset: RSDataset):
        if dataset.writer_mode == "none":
            writername = None
        else:
            writername = f"KVG(pad={self.padding},sw={self.stroke_width})"

        for charname in self.charnames:
            kvg = self.kvgcontainer.get_kvg_by_charname(charname)

            stack = [kvg]
            while len(stack):
                kvg = stack.pop()

                if self.mode == "character":
                    pass
                elif self.mode == "radical":
                    stack += kvg.children
                else:
                    raise Exception(f"unknown mode: {self.mode}")

                if not dataset.decomposer.is_kvgid_registered(kvg.kvgid):
                    continue

                assert kvg.name is not None
                
                imagepath = kvg.get_imagepath(image_size=dataset.image_size, padding=self.padding, stroke_width=self.stroke_width)

                radicallist = dataset.decomposer.get_decomposition_by_kvgid(kvg.kvgid)

                dataset.append_item(RawDatasetItem(charname=kvg.name, radicallist=radicallist, writername=writername, imagepath=imagepath))


class KvgCompositionDatasetProvider:
    kvgcontainer: Final[KvgContainer]
    composition_name: Final[str]

    padding: Final[int]
    stroke_width: Final[int]

    def __init__(
        self,

        kvg_path: str,
        composition_name: str,

        padding: int,
        stroke_width: int,

        n_limit: Optional[int],
    ):
        self.kvgcontainer = KvgContainer(kvg_path)
        self.composition_name = composition_name

        self.padding = padding
        self.stroke_width = stroke_width

        self.n_limit = n_limit

    @property
    def info(self):
        return {
            "kvg_path": self.kvgcontainer.kvg_path,
            "composition_name": self.composition_name,
        }
    
    def append_to(self, dataset: RSDataset):
        with open(pathstr(self.kvgcontainer.kvg_path, "output", "composition", f"{self.composition_name}.json")) as f:
            info = json.load(f)
            compositions = info["compositions"]
            
        if dataset.writer_mode == "none":
            writername = None
        else:
            writername = f"KVG_C({self.composition_name},pad={self.padding},sw={self.stroke_width})"

        for kvgids in compositions[:(self.n_limit or len(compositions))]:
            imagepaths: list[str] = []
            radicallist: list[Radical] = []

            for kvgid in kvgids:
                kvg = self.kvgcontainer.get_kvg_by_kvgid(kvgid)

                imagepaths.append(kvg.get_imagepath(dataset.image_size, self.padding, self.stroke_width))

                decomposition = dataset.decomposer.get_decomposition_by_kvgid(kvg.kvgid)
                assert len(decomposition) == 1
                radicallist.append(decomposition[0])

            dataset.append_item(RawDatasetItem(charname=",".join(kvgids), radicallist=radicallist, writername=writername, imagepath=tuple(imagepaths)))


class EtlcdbDatasetProvider:
    etlcdb_path: Final[str]
    etlcdb_name: Final[str]
    preprocess_type: Final[str]

    def __init__(self, etlcdb_path: str, etlcdb_name: str, preprocess_type: str):
        self.etlcdb_path = etlcdb_path
        self.etlcdb_name = etlcdb_name
        self.preprocess_type = preprocess_type

    @property
    def info(self) -> dict:
        return {
            "etlcdb_path": self.etlcdb_path,
            "etlcdb_name": self.etlcdb_name,
            "preprocess_type": self.preprocess_type,
        }

    def append_to(self, dataset: RSDataset):
        with open(pathstr(self.etlcdb_path, f"{self.etlcdb_name}.json")) as f:
            json_data = json.load(f)

        for item in json_data:
            rel_imagepath = item["Path"] # ex) ETL4/5001/0x3042.png
            abs_imagepath = pathstr(self.etlcdb_path, self.preprocess_type, rel_imagepath)

            charname = item["Character"] # ex) "あ"
            assert isinstance(charname, str)

            if dataset.writer_mode == "none":
                writername = None
            elif dataset.writer_mode == "dataset":
                writername = "ETLCDB"
            elif dataset.writer_mode == "all":
                serial_sheet_number = int(item["Serial Sheet Number"]) # ex) 5001
                writername = f"{self.etlcdb_name}_{serial_sheet_number}"
            else:
                raise Exception(f"unknown writer_mode: {dataset.writer_mode}")

            radicallist = dataset.decomposer.get_decomposition_by_charname(charname)

            dataset.append_item(RawDatasetItem(
                charname=charname,
                radicallist=radicallist,
                writername=writername,
                imagepath=abs_imagepath,
            ))


class RandomFontDatasetProvider:
    font_dataset_path: Final[str]

    font_name: Final[str]
    n_items: Final[int]

    def __init__(
        self,

        font_dataset_path: str,

        font_name: str,
        n_items: int,
    ):
        self.font_dataset_path = font_dataset_path

        self.font_name = font_name
        self.n_items = n_items

    @property
    def info(self):
        return {
            "font_dataset_path": self.font_dataset_path,

            "font_name": self.font_name,
            "n_items": self.n_items,
        }
    
    def append_to(self, dataset: RSDataset):
        if dataset.writer_mode != "none":
            raise NotImplementedError()

        parentpath = pathstr(self.font_dataset_path, f"{self.font_name} | random")
        imagepaths = os.listdir(parentpath)
        imagepaths.sort()
        if len(imagepaths) < self.n_items:
            raise Exception("there are not enough images")
        
        for imagepath in imagepaths[:self.n_items]:
            charcode = imagepath.split(" ")[0]
            charname = chr(int(charcode, base=16))

            radicallist = dataset.decomposer.get_decomposition_by_charname(charname)

            dataset.append_item(RawDatasetItem(
                charname=charname,
                radicallist=radicallist,
                writername=None,
                imagepath=pathstr(parentpath, imagepath),
            ))


class Kodomo2023DatasetProvider:
    kodomo2023_path: Final[str]
    process_type: str
    charnames: Final[set[str]]

    def __init__(self, kodomo2023_path: str, process_type: str, charnames: Iterable[str]):
        self.kodomo2023_path = kodomo2023_path
        self.process_type = process_type
        self.charnames = set(charnames)

    @property
    def info(self):
        return {
            "kodomo2023_path": self.kodomo2023_path,
            "process_type": self.process_type,
        }
    
    def append_to(self, dataset: RSDataset):
        with open(pathstr(self.kodomo2023_path, "kodomo_noclass.json")) as f:
            relimgpath2charname = json.load(f)
        
        train_images_root = pathstr(self.kodomo2023_path, "kodomo_charimgs_tt")
        if len(self.process_type):
            train_images_root += f" {self.process_type}"
        train_images_root = pathstr(train_images_root, "train")
        
        for abs_imagepath in glob.iglob(pathstr(train_images_root, "**/*.png"), recursive=True):
            rel_imagepath = os.path.relpath(abs_imagepath, train_images_root)

            # 不要な画像は除いてある (?)
            if rel_imagepath not in relimgpath2charname:
                continue

            charname = relimgpath2charname[rel_imagepath]
            if charname not in self.charnames:
                continue

            radicallist = dataset.decomposer.get_decomposition_by_charname(charname)

            if dataset.writer_mode == "none":
                writername = None
            elif dataset.writer_mode == "dataset":
                writername = "kodomo2023"
            elif dataset.writer_mode == "all":
                writername = f"kodomo2023({rel_imagepath.split('/')[1]})"
            else:
                raise Exception(f"unknown writer_mode: {dataset.writer_mode}")
            
            dataset.append_item(RawDatasetItem(
                charname=charname,
                radicallist=radicallist,
                writername=writername,
                imagepath=abs_imagepath,
            ))


class RawDatasetItem(NamedTuple):
    charname: str
    radicallist: list[Radical]
    writername: Optional[str]
    imagepath: Union[str, tuple[str, ...]]


class DatasetItem(NamedTuple):
    image: torch.Tensor
    radicallist: list[Radical]
    writeridx: Optional[int]


class DataloaderItem(NamedTuple):
    images: torch.Tensor
    radicallists: list[list[Radical]]
    writerindices: Optional[list[int]]


class RSDataset(Dataset):
    decomposer: Final[CharacterDecomposer]
    writer_mode: Final[WriterMode]
    image_size: Final[int]

    items: Final[list[RawDatasetItem]]

    radicalname2idx: Final[dict[str, int]]
    writername2idx: Final[Optional[dict[str, int]]]

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
        self.writername2idx = None if writer_mode == "none" else {}

        self.info = {
            "decomposer": decomposer.info,
            "writer_mode": writer_mode,
            "image_size": image_size,
            "data": [],
        }
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> DatasetItem:
        item = self.items[idx]

        if isinstance(item.imagepath, str):
            image = Image.open(item.imagepath).convert("RGB")
        else:
            image = compose_L_images(tuple(convert_to_L(Image.open(p)) for p in item.imagepath)).image.convert("RGB")
        
        image = TVF.to_tensor(image)

        radicallist = copy.deepcopy(item.radicallist)
        for radical in radicallist:
            radical.set_idx(self.radicalname2idx)
        
        if self.writername2idx is None:
            assert item.writername is None
            writeridx = None
        else:
            assert item.writername is not None
            writeridx = self.writername2idx[item.writername]
        
        return DatasetItem(image, radicallist, writeridx)
    
    def append_item(self, item: RawDatasetItem):
        for radical in item.radicallist:
            self.radicalname2idx.setdefault(radical.name, len(self.radicalname2idx))
        
        if self.writer_mode == "none":
            if item.writername is not None:
                raise Exception("item.writername must be None")
        else:
            if item.writername is None:
                raise Exception("item.writername cannot be None")
            assert self.writername2idx is not None
            self.writername2idx.setdefault(item.writername, len(self.writername2idx))

        self.items.append(item)

        # validation
        appended = self[len(self) - 1]
        assert appended.image.size() == (3, self.image_size, self.image_size)

    def append_by_provider(self, provider: DatasetProvider):
        p = len(self)
        provider.append_to(self)
        self.info["data"].append({
            "class": provider.__class__.__name__,
            "size": len(self) - p,
            "info": provider.info,
        })

    def create_dataloader(
        self,
        batch_size,
        shuffle,
        num_workers,
        shuffle_radicallist_of_char,
    )-> DataLoader[DataloaderItem]:
        def collate_fn(batch: list[DatasetItem]) -> DataloaderItem:
            nonlocal shuffle_radicallist_of_char

            images: torch.Tensor = torch.stack([item.image for item in batch])

            radicallists: list[list[Radical]] = [item.radicallist for item in batch]
            if shuffle_radicallist_of_char:
                for radicallist in radicallists:
                    random.shuffle(radicallist)
            
            writerindices: Optional[list[int]] = None
            if self.writername2idx is not None:
                writerindices = [0 for _ in batch]
                for i, item in enumerate(batch):
                    assert item.writeridx is not None
                    writerindices[i] = item.writeridx

            return DataloaderItem(images, radicallists, writerindices)
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


DatasetProvider = Union[KvgDatasetProvider, KvgCompositionDatasetProvider, EtlcdbDatasetProvider, RandomFontDatasetProvider, Kodomo2023DatasetProvider]
CharacterDecomposer = Union[BoundingBoxDecomposer, ClusteringLabelDecomposer, IdentityDecomposer]
WriterMode = Union[Literal["none"], Literal["dataset"], Literal["all"]]
