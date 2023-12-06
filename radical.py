from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional, Union


@dataclass
class Radical:
    name: Final[str]
    position: Final[Optional[Union[BoundingBox, ClusteringLabel]]]

    idx_: Optional[int] = None

    @property
    def idx(self) -> int:
        if self.idx_ is None:
            raise Exception("idx is not registered")
        return self.idx_

    def set_idx(self, radicalname2idx):
        self.idx_ = radicalname2idx[self.name]

    @staticmethod
    def from_dict(dct) -> Radical:
        position = dct.pop("position")
        if "left" in position:
            position = BoundingBox(**position)
        elif "label" in position:
            position = ClusteringLabel(**position)
        else:
            raise Exception()
        
        return Radical(position=position, **dct)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


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


@dataclass
class ClusteringLabel:
    label: int
