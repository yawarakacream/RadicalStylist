from argparse import ArgumentParser
from glob import glob
from typing import Union

import torch

from character_decomposer import IdentityDecomposer, BoundingBoxDecomposer, ClusteringLabelDecomposer
from dataset import CharacterDecomposer, Radical
from radical import Radical
from radical_stylist import RadicalStylist
from train import prepare_radicallists_with_name
from utility import pathstr, char2code, save_images


def main(
    save_path: str,
    model_name:str,

    character_decomposer: CharacterDecomposer,
    chars: list[Union[str, tuple[str, list[Radical]]]],
    writers: Union[int, list[str]],
    
    device: torch.device,
):
    print(f"save_path: {save_path}")
    print(f"device: {device}")

    print("loading RadicalStylist...")
    rs = RadicalStylist.load(save_path=save_path, model_name=model_name, device=device)
    print("loaded.")

    radicallists_with_name = prepare_radicallists_with_name(character_decomposer, rs.radicalname2idx, chars)
    for name, radicallist in radicallists_with_name:
        print(f"\t{name} = {' + '.join(map(lambda r: r.name, radicallist))}")
    
    radicallists = [r for _, r in radicallists_with_name]

    images_list = rs.sample(radicallists, writers)

    save_directory = pathstr(rs.save_path, "generated")

    for i, ((name, radicallist), images) in enumerate(zip(radicallists_with_name, images_list)):
        if isinstance(writers, int):
            path = f"trained_{char2code(name or '?')}"
        elif isinstance(writers, list):
            path = f"trained_{char2code(name or '?')}_{writers[i]}"

        path = pathstr(save_directory, path)

        c = len(glob(f"{path}*.png"))
        if 0 < c:
            path += f" ({c})"
        
        path += ".png"

        save_images(images, path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(
        save_path=pathstr("./output/character384 nlp2024(kana,kanji)"),
        model_name="checkpoint=0100",

        character_decomposer=IdentityDecomposer(),
        chars=list("あかさたな"),
        writers=[
            *["kodomo2023" for _ in range(8)],
            *["ETLCDB" for _ in range(8)]
        ],

        device=torch.device(args.device),
    )
