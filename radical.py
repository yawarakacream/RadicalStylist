from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from kanjivg import Kvg


@dataclass
class Radical:
    name: Final[str]
    idx_: Optional[int]

    left: Final[float]
    right: Final[float]
    top: Final[float]
    bottom: Final[float]

    children: Final[list[Radical]]

    def __init__(self, name, **kwargs):
        self.name = name
        self.idx_ = kwargs.get("idx", None)

        if "left" in kwargs:
            self.left = kwargs["left"]
            self.right = kwargs["right"]
            self.top = kwargs["top"]
            self.bottom = kwargs["bottom"]
        
        elif "center_x" in kwargs:
            center_x = kwargs["center_x"]
            center_y = kwargs["center_y"]
            width = kwargs["width"]
            height = kwargs["height"]

            self.left = center_x - width / 2
            self.right = center_x + width / 2
            self.top = center_y - height / 2
            self.bottom = center_y + height/ 2
        
        else:
            raise Exception(f"invalid arguments: {kwargs}")
    
        self.children = kwargs.get("children", [])

    @staticmethod
    def from_kvg(kvg: Kvg, kvg_path: str, image_size=64, line_width=1, image_path: Optional[str] = None):
        from PIL import Image

        name = kvg.name
        if name is None:
            raise Exception("cannot use nameless radical")
        if kvg.part is not None:
            name = f"{name}_{kvg.part}"
        
        if image_path is None:
            image_path = kvg.get_image_path(kvg_path, image_size=image_size, line_width=line_width)

        image = np.array(Image.open(image_path).convert("1")).transpose()
        nonzero_idx = image.nonzero()
        nonzero_idx[0].sort()
        nonzero_idx[1].sort()
        
        left = (nonzero_idx[0][0] - 1) / image_size
        right = (nonzero_idx[0][-1]) / image_size
        top = (nonzero_idx[1][0] - 1) / image_size
        bottom = (nonzero_idx[1][-1]) / image_size

        children = []
        if len(kvg.svg) == 0: # 角がある部首は子部首に分解しない
            stack = list(reversed(kvg.children))
            while len(stack):
                kvg0 = stack.pop()
                if kvg0.name is None:
                    stack += kvg0.children
                else:
                    children.append(Radical.from_kvg(kvg0, kvg_path, image_size, line_width))

        return Radical(name=name, left=left, right=right, top=top, bottom=bottom, children=children)

    @staticmethod
    def from_dict(dct):
        return Radical(**dct)

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @property
    def idx(self) -> int:
        if self.idx_ is None:
            raise Exception("idx is not registered")
        return self.idx_

    def set_idx(self, radicalname2idx):
        self.idx_ = radicalname2idx[self.name]
        # for r in self.children:
        #     r.set_idx(radicalname2idx)
        
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
    
    def get_radicals(self, depth: str) -> list[Radical]:
        if depth == "zero":
            return [self]
        
        elif depth == "max":
            if len(self.children) == 0:
                return [copy.deepcopy(self)]

            ret = []
            for c in self.children:
                for r in c.get_radicals("max"):
                    ret.append(copy.deepcopy(r))
            return ret

        else:
            raise Exception(f"unknown depth: {depth}")
