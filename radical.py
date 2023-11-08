from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Final, Optional


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
    def from_radicaljson(json):
        name = json["name"]
        if name is None:
            raise Exception(f"name is None: {json}")

        if json["part"] is not None:
            name += f'_{json["part"]}'

        bounding = json["bounding"]

        left = bounding["left"]
        right = bounding["right"]
        top = bounding["top"]
        bottom = bounding["bottom"]

        children = []
        for child_json in json["children"]:
            queue = [child_json]
            while len(queue):
                el = queue.pop()
                if el["name"] is None:
                    queue += el["children"]
                else:
                    children.append(Radical.from_radicaljson(el))

        return Radical(name, left=left, right=right, top=top, bottom=bottom, children=children)

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
        for r in self.children:
            r.set_idx(radicalname2idx)
        
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
            
        elif depth == "binary-random":
            if random.random() < 0.5:
                return [copy.deepcopy(self)]

            ret = []
            for c in self.children:
                for r in c.get_radicals("binary-random"):
                    ret.append(copy.deepcopy(r))
            return ret

        elif depth == "legacy":
            if len(self.children) < 1:
                return [copy.deepcopy(self)]

            ret = []
            for c in self.children:
                for r in c.get_radicals("max"):
                    ret.append(copy.deepcopy(r))
            return ret
            
        else:
            raise Exception(f"unknown depth: {depth}")
