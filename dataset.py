import copy
import json
import random
from typing import Any, Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader

import character_utility
from radical import Radical
from utility import pathstr, read_image_as_tensor


class RSDataset(Dataset):
    charname2radicaljson: dict[str, list[Radical]]
    ignore_kana: bool
    radical_depth: str

    items: list[tuple[str, Radical, str]] # (image_path, rootradical, writername)

    info: dict[str, Any]

    def __init__(
        self,

        charname2radicaljson,
        radical_depth="max",
        ignore_kana=False,

        items=None,

        info=None,
    ):
        self.charname2radicaljson = charname2radicaljson
        self.ignore_kana = ignore_kana
        self.radical_depth = radical_depth

        self.items = items or []

        self.info = info or {
            "ignore_kana": ignore_kana,
            "datasets": [],
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> tuple[str, list[Radical], str]:
        image_path, rootradical, writerindices = self.items[idx]
        radicals = rootradical.get_radicals(self.radical_depth)
        return (image_path, radicals, writerindices)

    def add_item(self, image_path: str, rootradicalname: str, writername: str) -> None:
        if self.ignore_kana and (rootradicalname in character_utility.all_kanas):
            return

        rootradical = Radical.from_radicaljson(self.charname2radicaljson[rootradicalname])

        self.items.append((image_path, rootradical, writername))

    def add_etlcdb(self, etlcdb_path: str, etlcdb_process_type: str, etlcdb_name: str) -> None:
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

    def add_from_corpuses_string(self, corpuses, etlcdb_path: str) -> None:
        for corpus in corpuses:
            corpus = corpus.split("/")
            corpus_type = corpus[0]

            if corpus_type == "etlcdb":
                _, etlcdb_process_type, etlcdb_names = corpus
                etlcdb_names = etlcdb_names.split(",")

                for etlcdb_name in etlcdb_names:
                    self.add_etlcdb(etlcdb_path, etlcdb_process_type, etlcdb_name)

            else:
                raise Exception(f"unknown corpus type: {corpus_type}")

    def create_radicalname2idx(self) -> dict[str, int]:
        radicalname2idx: dict[str, int] = {}

        if self.radical_depth == "binary-random":
            queue: list[Radical] = []
            for _, radical, _ in self.items:
                for r in radical.get_radicals(self.radical_depth):
                    queue.append(r)
            
            while len(queue):
                r = queue.pop()
                radicalname2idx.setdefault(r.name, len(radicalname2idx))
                queue += r.children

        else:
            for _, radical, _ in self.items:
                for r in radical.get_radicals(self.radical_depth):
                    radicalname2idx.setdefault(r.name, len(radicalname2idx))

        return radicalname2idx

    def create_writername2idx(self) -> dict[str, int]:
        writername2idx: dict[str, int] = {}
        for _, _, writername in self.items:
            writername2idx.setdefault(writername, len(writername2idx))
        return writername2idx

    def random_split(self, sizes: Sequence[float]):
        lengths = [int(len(self.items) * l) for l in sizes]
        assert all(map(lambda l: 0 <= l, lengths))

        r = len(self.items) - sum(lengths)
        assert 0 <= r

        i = 0
        while 0 < r:
            lengths[i % len(lengths)] += 1
            i += 1
            r -= 1

        items = copy.deepcopy(self.items)
        random.shuffle(items)

        sub_list: list[RSDataset] = []
        a = 0
        for l in lengths:
            sub_list.append(RSDataset(
                charname2radicaljson=self.charname2radicaljson,
                ignore_kana=self.ignore_kana,

                items=items[a:(a + l)],

                info=self.info,
            ))
            a += l

        return sub_list


def create_dataloader(
    dataset,
    batch_size,
    shuffle_dataset,
    num_workers,
    shuffle_radicals_of_char,
    radicalname2idx,
    writername2idx,
)-> DataLoader[tuple[torch.Tensor, list[Radical], Optional[list[int]]]]:
    assert isinstance(dataset, RSDataset)

    def collate_fn(batch: list[tuple[str, list[Radical], str]]) -> tuple[torch.Tensor, list[Radical], Optional[list[int]]]:
        nonlocal shuffle_radicals_of_char, radicalname2idx, writername2idx

        images: Any = [None for _ in range(len(batch))]
        radicallists: Any = [None for _ in range(len(batch))]
        writerindices = writername2idx and [None for _ in range(len(batch))]

        for i, (image_path, radicals, writername) in enumerate(batch):
            images[i] = read_image_as_tensor(image_path)

            radicallists[i] = copy.deepcopy(radicals)
            for r in radicallists[i]:
                r.set_idx(radicalname2idx)
            if shuffle_radicals_of_char:
                random.shuffle(radicallists[i])

            if writerindices is not None:
                writerindices[i] = writername2idx[writername]

        images = torch.stack(images, 0)

        return images, radicallists, writerindices

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader
