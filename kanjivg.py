from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Optional

from utility import pathstr


@dataclass
class Kvg:
    kvgid: str
    name: Optional[str]
    part: Optional[str]
    position: Optional[str]
    svg: list[str]
    children: list[Kvg]
    
    @staticmethod
    def from_dict(dct):
        children = dct.pop("children")
        children = [Kvg.from_dict(c) for c in children]
        return Kvg(children=children, **dct)
    
    def get_directory_path(self, kvg_path: str):
        charcode = self.kvgid.split("-")[0]
        return pathstr(kvg_path, "build", charcode[:-2] + "00", charcode)

    def get_image_path(self, kvg_path: str, image_size: int, line_width: int):
        return pathstr(
            self.get_directory_path(kvg_path),
            f"{image_size}x lw={line_width} {self.kvgid}.png",
        )

class KvgContainer:
    kvg_path: str

    def __init__(self, kvg_path: str):
        self.kvg_path = kvg_path
    
    def get_kvg(self, charname):
        charcode = ord(charname)
        parentcode_str = format((charcode >> 8) << 8, "#07x")[len("0x"):]
        charcode_str = format(charcode, "#07x")[len("0x"):]
        path = pathstr(self.kvg_path, "build", parentcode_str, charcode_str, f"{charcode_str}.json")
        with open(path) as f:
            dct = json.load(f)
        return Kvg.from_dict(dct)
    