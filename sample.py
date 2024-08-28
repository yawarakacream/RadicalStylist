from argparse import ArgumentParser
from glob import glob
from typing import Union

import torch

from character_decomposer import BoundingBoxDecomposer, ClusteringLabelDecomposer
from dataset import CharacterDecomposer, Radical
from radical import Radical
from radical_stylist import RadicalStylist
from train import prepare_radicallists_with_name
from utility import pathstr, char2code, save_images


def main(
    save_path: str,

    character_decomposer: CharacterDecomposer,
    chars: list[Union[str, tuple[str, list[Radical]]]],
    writers: Union[int, list[str]],
    
    device: torch.device,
):
    print(f"save_path: {save_path}")
    print(f"device: {device}")

    print("loading RadicalStylist...")
    rs = RadicalStylist.load(save_path=save_path, model_name="checkpoint=0100", device=device)
    print("loaded.")

    save_directory = pathstr(rs.save_path, "generated")

    radicallists_with_name = prepare_radicallists_with_name(character_decomposer, rs.radicalname2idx, chars)
    for name, radicallist in radicallists_with_name:
        print(f"\t{name} = {' + '.join(map(lambda r: r.name, radicallist))}")
    
    radicallists = [r for _, r in radicallists_with_name]

    images_list = rs.sample(radicallists, writers)

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

    # chars 8 * writer 224: 約 30 分
    main(
        save_path=pathstr("./output/nlp2024/nlp2024+KVG(pad={4,8,12,16},sw=2) radenc=cl_0"),

        character_decomposer=ClusteringLabelDecomposer(
            kvg_path=pathstr("~/datadisk/dataset/kanjivg"),
            radical_clustering_name="edu+jis_l1,2 n_clusters=384 (imsize=16,sw=2,blur=2)",
        ),
        chars=[
            *"日谷一人分想長点",
        ],
        writers=[
            *["kodomo2023" for _ in range(224)],
        ],

        device=torch.device(args.device),
    )
