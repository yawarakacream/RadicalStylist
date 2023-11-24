from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np

from PIL import Image

from dataset import Radical
from kanjivg import Kvg, KvgContainer


@dataclass
class BoundingBox:
    left: float
    right: float
    top: float
    bottom: float

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def width(self) -> float:
        return self.right - self.left
    
    @property
    def height(self) -> float:
        return self.bottom - self.top


class BoundingBoxDecomposer:
    depth: Literal["max"]
    image_size: int
    padding: int
    stroke_width: int

    kvgcontainer: KvgContainer
    kvgid2radicallist: Optional[dict[str, list[Radical]]]
    charname2kvgid: Optional[dict[str, str]]

    def __init__(self, depth: Literal["max"], image_size: int, padding: int, stroke_width: int):
        self.depth = depth
        self.image_size = image_size
        self.padding = padding
        self.stroke_width = stroke_width

    @property
    def name(self):
        return "bounding-box"
    
    @property
    def info(self) -> dict:
        return {
            "name": "bounding-box",
            "depth": self.depth,
            "image_size": self.image_size,
            "padding": self.padding,
            "stroke_width": self.stroke_width,
        }

    def init(self, kvgcontainer: KvgContainer, charnames: Iterable[str]):
        self.kvgid2radicallist = {}
        self.charname2kvgid = {}

        self.kvgcontainer = kvgcontainer

        def dfs(kvg: Kvg) -> list[Radical]:
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

            assert self.kvgid2radicallist is not None
            self.kvgid2radicallist[kvg.kvgid] = ret

            return ret

        for charname in charnames:
            kvg = kvgcontainer.get_kvg(charname)
            self.charname2kvgid[charname] = kvg.kvgid
            dfs(kvg)

    def is_kvgid_registered(self, kvgid: str) -> bool:
        assert self.kvgid2radicallist is not None
        return kvgid in self.kvgid2radicallist

    def get_decomposition_by_kvgid(self, kvgid: str) -> list[Radical]:
        assert self.kvgid2radicallist is not None
        return self.kvgid2radicallist[kvgid]

    def get_decomposition_by_charname(self, charname: str) -> list[Radical]:
        assert self.charname2kvgid is not None
        return self.get_decomposition_by_kvgid(self.charname2kvgid[charname])
