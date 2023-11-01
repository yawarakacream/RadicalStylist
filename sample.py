import argparse
from glob import glob
from typing import Union

import torch

from character import Char, Radical
from image_vae import StableDiffusionVae
from radical_stylist import RadicalStylist
from utility import pathstr, save_images, char2code, create_charname2radicaljson


def prepare_chars(chars: list[Union[str, Char]], charname2radicaljson: dict[str, Char], radicalname2idx: dict[str, int]) -> list[Char]:
    ret = []
    for char in chars:
        if isinstance(char, str):
            ret.append(Char.from_radicaljson(charname2radicaljson[char]))
        elif isinstance(char, Char):
            ret.append(char)
        else:
            raise Exception()
        
        ret[-1].register_radicalidx(radicalname2idx)
        
        print("\t" + ret[-1].to_formula_string())
    
    return ret


def main(
    save_path: str,
    stable_diffusion_path: str,
    radicals_data_path: str,
    
    chars: list[Union[str, Char]],
    writers: Union[int, list[str]],
    
    device: torch.device,
):
    print(f"save_path: {save_path}")
    
    charname2radicaljson = create_charname2radicaljson(radicals_data_path)

    vae = StableDiffusionVae(stable_diffusion_path, device)

    print("loading RadicalStylist...")
    rs = RadicalStylist.load(save_path=save_path, vae=vae, device=device)
    print("loaded.")

    prepared_chars = prepare_chars(chars, charname2radicaljson, rs.radicalname2idx)
    images_list = rs.sample(prepared_chars, writers)

    save_directory = pathstr(rs.save_path, "generated")

    for i, (char, images) in enumerate(zip(prepared_chars, images_list)):
        if isinstance(writers, int):
            path = f"trained_{char2code(char.name or '?')}"
        elif isinstance(writers, list):
            path = f"trained_{char2code(char.name or '?')}_{writers[i]}"

        path = pathstr(save_directory, path)

        c = len(glob(f"{path}*.png"))
        if 0 < c:
            path += f" ({c})"
        
        path += ".png"

        save_images(images, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    main(
        save_path=pathstr("./output/rs ignore_writer ETL8G_400"),
        stable_diffusion_path=pathstr("~/datadisk/stable-diffusion-v1-5"),
        radicals_data_path=pathstr("~/datadisk/dataset/kanjivg/build/all.json"),

        chars=[
            # 訓練データにある字
            *"何標園遠",

            # 部首単体
            Char("亻", [Radical("亻", left=0.078125, right=0.3125, top=0.140625, bottom=0.875)]),
            Char("宀", [Radical("宀", left=0.140625, right=0.859375, top=0.078125, bottom=0.375)]),
            Char("广", [Radical("广", left=0.078125, right=0.78125, top=0.078125, bottom=0.84375)]),
            Char("⻌", [Radical("⻌", left=0.109375, right=0.8125, top=0.15625, bottom=0.828125)]),

            # 訓練データにないが部首データにある字
            *"倹困麻諭",
        ],
        # writers=[f"ETL8G_400_{i}" for i in range(1, 8 + 1)],
        writers=8,

        device=torch.device(args.device),
    )
