import json
from typing import Final, Literal

import numpy as np

from PIL import Image

from kanjivg import Kvg, KvgContainer
from radical import BoundingBox, ClusteringLabel, Radical
from utility import pathstr


class BoundingBoxDecomposer:
    kvgcontainer: Final[KvgContainer]
    depth: Final[Literal["max"]]
    image_size: Final[int]
    padding: Final[int]
    stroke_width: Final[int]

    charname2kvgid: Final[dict[str, str]]
    kvgid2radicallist: Final[dict[str, list[Radical]]]

    def __init__(self, kvg_path: str, depth: Literal["max"], image_size: int, padding: int, stroke_width: int):
        self.kvgcontainer = KvgContainer(kvg_path)
        self.depth = depth
        self.image_size = image_size
        self.padding = padding
        self.stroke_width = stroke_width

        self.charname2kvgid = {}
        self.kvgid2radicallist = {}

    @property
    def name(self) -> str:
        return "bounding-box"
    
    @property
    def info(self) -> dict:
        return {
            "name": self.name,
            "kvg_path": self.kvgcontainer.kvg_path,
            "depth": self.depth,
            "image_size": self.image_size,
            "padding": self.padding,
            "stroke_width": self.stroke_width,
        }

    def register(self, charname: str):
        def dfs(kvg: Kvg) -> list[Radical]:
            assert kvg.kvgid not in self.kvgid2radicallist
            
            ret: list[Radical] = []
            if len(kvg.svg) == 0:
                for kvg0 in kvg.children:
                    ret0 = dfs(kvg0)
                    if len(ret0) == 0:
                        ret.clear()
                        break
                    ret += ret0

            if kvg.name is None:
                return ret
            
            if len(ret) == 0:
                name = kvg.name
                if kvg.part is not None:
                    name = f"{name}_{kvg.part}"
                
                image_path = kvg.get_image_path(image_size=self.image_size, padding=self.padding, stroke_width=self.stroke_width)
                image = np.array(Image.open(image_path).convert("1")).transpose()
                nonzero_idx = image.nonzero()
                nonzero_idx[0].sort()
                nonzero_idx[1].sort()
                
                left = (nonzero_idx[0][0] - 1) / self.image_size
                right = (nonzero_idx[0][-1]) / self.image_size
                top = (nonzero_idx[1][0] - 1) / self.image_size
                bottom = (nonzero_idx[1][-1]) / self.image_size

                ret.append(Radical(
                    name=name,
                    position=BoundingBox(
                        left=left,
                        right=right,
                        top=top,
                        bottom=bottom,
                    ),
                ))

            self.kvgid2radicallist[kvg.kvgid] = ret
            return ret
        
        if charname in self.charname2kvgid:
            return

        kvg = self.kvgcontainer.get_kvg(charname)

        self.charname2kvgid[charname] = kvg.kvgid
        dfs(kvg)

    def is_kvgid_registered(self, kvgid: str) -> bool:
        return kvgid in self.kvgid2radicallist

    def get_decomposition_by_kvgid(self, kvgid: str) -> list[Radical]:
        return self.kvgid2radicallist[kvgid]

    def get_decomposition_by_charname(self, charname: str) -> list[Radical]:
        return self.get_decomposition_by_kvgid(self.charname2kvgid[charname])


class ClusteringLabelDecomposer:
    kvgcontainer: Final[KvgContainer]
    radical_clustering_path: Final[str]

    kvgid2label: Final[dict[str, int]]
    decompositions: Final[dict[str, list[str]]]

    charname2kvgid: Final[dict[str, str]]
    kvgid2radicallist: Final[dict[str, list[Radical]]]

    def __init__(self, kvg_path: str, radical_clustering_path: str):
        self.kvgcontainer = KvgContainer(kvg_path)
        self.radical_clustering_path = radical_clustering_path

        with open(pathstr(self.radical_clustering_path, "label2radical2kvgids.json")) as f:
            label2radical2kvgids: list[dict[str, list[str]]] = json.load(f)

        self.kvgid2label = {}
        for label, radical2kvgids in enumerate(label2radical2kvgids):
            for kvgids in radical2kvgids.values():
                for kvgid in kvgids:
                    self.kvgid2label[kvgid] = label
        
        with open(pathstr(self.radical_clustering_path, "decompositions.json")) as f:
            self.decompositions = json.load(f)

        self.charname2kvgid = {}
        self.kvgid2radicallist = {}

    @property
    def name(self) -> str:
        return "clustering-label"
    
    @property
    def info(self) -> dict:
        return {
            "name": self.name,
            "kvg_path": self.kvgcontainer.kvg_path,
            "radical_clustering_path": self.radical_clustering_path,
        }

    def register(self, charname: str):
        kvg = self.kvgcontainer.get_kvg(charname)
        if charname in self.charname2kvgid:
            return

        self.charname2kvgid[charname] = kvg.kvgid

        kvgid2radicalname: dict[str, str] = {}
        stack = [kvg]
        while len(stack):
            kvg0 = stack.pop()
            stack += kvg0.children

            if kvg0.name is not None:
                kvgid2radicalname[kvg0.kvgid] = kvg0.name
        
        stack = [kvg]
        while len(stack):
            kvg0 = stack.pop()
            stack += kvg0.children

            if kvg0.kvgid not in self.decompositions:
                continue

            radicallist: list[Radical] = [
                Radical(
                    name=kvgid2radicalname[ckvgid0],
                    position=ClusteringLabel(label=self.kvgid2label[ckvgid0])
                )
                for ckvgid0 in self.decompositions[kvg0.kvgid]
            ]
            self.kvgid2radicallist[kvg0.kvgid] = radicallist

    def is_kvgid_registered(self, kvgid: str) -> bool:
        return kvgid in self.kvgid2radicallist

    def get_decomposition_by_kvgid(self, kvgid: str) -> list[Radical]:
        return self.kvgid2radicallist[kvgid]

    def get_decomposition_by_charname(self, charname: str) -> list[Radical]:
        return self.get_decomposition_by_kvgid(self.charname2kvgid[charname])
