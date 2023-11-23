from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Final, Optional

from utility import pathstr


@dataclass
class Kvg:
    kvgid: str
    name: Optional[str]
    part: Optional[str]
    position: Optional[str]
    svg: list[str]
    children: list[Kvg]

    container: Final[KvgContainer]
    
    @staticmethod
    def from_dict(container: KvgContainer, dct):
        children = dct.pop("children")
        children = [Kvg.from_dict(container, c) for c in children]
        return Kvg(container=container, children=children, **dct)
    
    @property
    def charcode(self):
        return self.kvgid.split("-")[0]
    
    @property
    def directory_path(self):
        return pathstr(self.container.kvg_path, "output", self.charcode[:-2] + "00", self.charcode)

    def get_image_path(self, image_size: int, padding: int, stroke_width: int):
        return pathstr(
            self.directory_path,
            f"{image_size}x,pad={padding},sw={stroke_width} {self.kvgid}.png",
        )


class KvgContainer:
    kvg_path: str

    def __init__(self, kvg_path: str):
        self.kvg_path = kvg_path
    
    def get_kvg(self, charname):
        charcode = format(ord(charname), "#07x")[len("0x"):]
        directory_path = pathstr(self.kvg_path, "output", charcode[:-2] + "00", charcode)
        json_path = pathstr(directory_path, f"{charcode}.json")
        with open(json_path) as f:
            dct = json.load(f)
        return Kvg.from_dict(self, dct)
    