from __future__ import annotations

import json
from copy import deepcopy

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
    def directory_path(self):
        root_kvgid = get_root_kvgid(self.kvgid)
        return pathstr(self.container.kvg_path, "output", "main", root_kvgid[:-2] + "00", root_kvgid)

    def get_image_path(self, image_size: int, padding: int, stroke_width: int):
        return pathstr(
            self.directory_path,
            f"{image_size}x,pad={padding},sw={stroke_width} {self.kvgid}.png",
        )


kvg_container_cache: dict[str, dict[str, Kvg]] = {} # {kvg_path: {kvgid: Kvg}}


class KvgContainer:
    kvg_path: str

    def __init__(self, kvg_path: str):
        self.kvg_path = kvg_path
    
    def get_kvg_by_charname(self, charname):
        kvgid = charname2kvgid(charname)
        return self.get_kvg_by_kvgid(kvgid)
    
    def get_kvg_by_kvgid(self, kvgid):
        root_kvgid = get_root_kvgid(kvgid)

        kvg_container_cache.setdefault(self.kvg_path, {})

        if root_kvgid not in kvg_container_cache[self.kvg_path]:
            directory_path = pathstr(self.kvg_path, "output", "main", root_kvgid[:-2] + "00", root_kvgid)
            json_path = pathstr(directory_path, f"{root_kvgid}.json")
            with open(json_path) as f:
                dct = json.load(f)

            root_kvg = Kvg.from_dict(self, dct)

            stack = [root_kvg]
            while len(stack):
                kvg = stack.pop()
                stack += kvg.children
                kvg_container_cache[self.kvg_path][kvg.kvgid] = deepcopy(kvg)

        return deepcopy(kvg_container_cache[self.kvg_path][kvgid])


def charname2kvgid(charname: str):
    return format(ord(charname), "#07x")[len("0x"):]


def get_root_kvgid(kvgid: str):
    return kvgid.split("-")[0]
