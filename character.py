from dataclasses import dataclass, asdict as dc_asdict
from typing import Optional


@dataclass
class Radical:
    name: str
    idx_: Optional[int]
    center_x: float
    center_y: float
    width: float
    height: float
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.idx = kwargs.get("idx", None)
        
        if "center_x" in kwargs:
            self.center_x = kwargs["center_x"]
            self.center_y = kwargs["center_y"]
            self.width = kwargs["width"]
            self.height = kwargs["height"]
            
        elif "left" in kwargs:
            left = kwargs["left"]
            right = kwargs["right"]
            top = kwargs["top"]
            bottom = kwargs["bottom"]
            
            self.center_x = (left + right) / 2
            self.center_y = (top + bottom) / 2
            self.width = right - left
            self.height = bottom - top
        
        else:
            raise Exception(f"invalid arguments: {kwargs}")
    
    def from_radicaljson(dct):
        name = dct["name"]
        if dct["part"] is not None:
            name += f'_{dct["part"]}'
        
        bounding = dct["bounding"]

        left = bounding["left"]
        right = bounding["right"]
        top = bounding["top"]
        bottom = bounding["bottom"]

        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        width = right - left
        height = bottom - top
        
        return Radical(name, center_x=center_x, center_y=center_y, width=width, height=height)
    
    def from_dict(dct):
        return Radical(**dct)
    
    def to_dict(self):
        return dc_asdict(self)
    
    @property
    def idx(self):
        if self.idx_ is None:
            raise Exception("idx is not registered")
        return self.idx_
    
    @idx.setter
    def idx(self, idx):
        self.idx_ = idx

    @property
    def left(self):
        return self.center_x - self.width / 2
    
    @property
    def right(self):
        return self.center_x + self.width / 2
    
    @property
    def top(self):
        return self.center_y - self.height / 2
    
    @property
    def bottom(self):
        return self.center_y + self.height / 2


@dataclass
class Char:
    name: Optional[str]
    radicals: list[Radical]
    
    def from_radicaljson(dct):
        def get_radicals(dct):
            from_children = []
            for d in dct["children"]:
                c = get_radicals(d)
                if c is None:
                    from_children = None
                    break
                from_children += c
            
            from_name = dct["name"] and [Radical.from_radicaljson(dct)]
            
            if (from_children is not None) and len(from_children):
                # 例) 三 = 三_1 + 三_2 = 一 + 三_2 のとき 三_1 + 三_2 を採用する
                if (len(from_children) == 1) and (from_name is not None):
                    return from_name
                
                return from_children
            
            return from_name
        
        name = dct["name"]
        radicals = get_radicals(dct)
        return Char(name, radicals)
    
    def from_dict(dct):
        name = dct["name"]
        radicals = [Radical.from_dict(r) for r in dct["radicals"]]
        return Char(name, radicals)
    
    def to_dict(self):
        name = self.name
        radicals = [r.to_dict() for r in self.radicals]
        return {"name": name, "radicals": radicals}
    
    def to_formula_string(self):
        return f"{self.name} = {' + '.join(map(lambda r: r.name, self.radicals))}"

    def register_radicalidx(self, radicalname2idx):
        for r in self.radicals:
            if r.name not in radicalname2idx:
                raise Exception(f"radicalname2idx does not support radical '{r.name}' of '{self.name}'")
            r.idx = radicalname2idx[r.name]
