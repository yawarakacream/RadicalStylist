import argparse
from typing import Iterable

import torch

from character_decomposer import IdentityDecomposer
from dataset import DatasetProvider, RSDataset, RandomFontDatasetProvider
from image_vae import StableDiffusionVae
from utility import pathstr


def main(
    *,

    save_path: str,
    original_vae_path: str,
    
    image_size,

    batch_size: int,
    epochs: int,
    shuffle_dataset: bool,
    dataloader_num_workers: int,

    datasets: Iterable[DatasetProvider],
    
    test_image_paths: list[str],

    device: torch.device,
):
    print(f"save_path: {save_path}")
    print(f"device: {device}")

    # train data
    print(f"train data:")

    dataset = RSDataset(decomposer=IdentityDecomposer(), writer_mode="none", image_size=image_size)
    print("\tprovider:")
    for provider in datasets:
        print(f"\t\t{provider.__class__.__name__}: ", end="", flush=True)
        l = len(dataset)
        dataset.append_by_provider(provider)
        print(len(dataset) - l)

    print(f"\ttotal data size: {len(dataset)}")

    dataloader = dataset.create_dataloader(
        batch_size=batch_size,
        shuffle=shuffle_dataset,
        num_workers=dataloader_num_workers,
        shuffle_radicallist_of_char=False,
    )

    print(f"test images:", "\n\t".join(test_image_paths), sep="\n")

    vae = StableDiffusionVae(original_vae_path).to(device=device)

    print("training started!")
    vae.train(save_path, dataloader, epochs, test_image_paths)
    print("training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    main(
        # save_path=pathstr("./output/vae/SanariFont001(n_random=65536)"),
        save_path=pathstr("./output/vae/SanariFont001(n_items=262140)"),
        original_vae_path=pathstr("~/datadisk/stable-diffusion-v1-5/vae"),

        image_size=64,

        batch_size=64,
        epochs=100,
        shuffle_dataset=True,
        dataloader_num_workers=4,

        # datasets=[
        #     RandomFontDatasetProvider(
        #         font_dataset_path=pathstr("~/datadisk/dataset/font"),
        #         font_name="Noto Sans JP Regular",
        #         n_random=65536,
        #     ),
        #     RandomFontDatasetProvider(
        #         font_dataset_path=pathstr("~/datadisk/dataset/font"),
        #         font_name="Noto Serif JP Regular",
        #         n_random=65536,
        #     ),
        # ],
        # datasets=[
        #     RandomFontDatasetProvider(
        #         font_dataset_path=pathstr("~/datadisk/dataset/font"),
        #         font_name="SanariFont SanariFont Light",
        #         n_items=65536,
        #     ),
        # ],
        datasets=[
            RandomFontDatasetProvider(
                font_dataset_path=pathstr("~/datadisk/dataset/font"),
                font_name="SanariFont SanariFont Light",
                n_items=262140,
            ),
        ],

        test_image_paths=[
            # ETL4 あ
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL4/5001/000000.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL4/5002/000051.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL4/5001/000000.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL4/5002/000051.png"),

            # ETL4 ぬ
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL4/5003/000124.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL4/5004/000175.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL4/5003/000124.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL4/5004/000175.png"),

            # ETL9B 何
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/2/607489.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/22/003325.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/42/006361.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/62/009397.png"),

            # ETL9B 標
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/17/609644.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/37/005480.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/57/008516.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/77/011552.png"),

            # ETL8G 何
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/1/000007.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/11/000963.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/1/000007.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/11/000963.png"),

            # ETL8G 標
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/7/000653.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/17/001609.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/7/000653.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/17/001609.png"),

            # ETL8G 遠
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/5/000389.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/15/001345.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/5/000389.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/15/001345.png"),

            # ETL8G 園
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/2/000102.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/12/001058.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/2/000102.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/12/001058.png"),

            # KVG 何
            pathstr("~/datadisk/dataset/kanjivg/output/main/04f00/04f55/64x,pad=4,sw=2 04f55.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/04f00/04f55/64x,pad=8,sw=2 04f55.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/04f00/04f55/64x,pad=12,sw=2 04f55.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/04f00/04f55/64x,pad=16,sw=2 04f55.png"),

            # KVG 何
            pathstr("~/datadisk/dataset/kanjivg/output/main/06a00/06a19/64x,pad=4,sw=2 06a19.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/06a00/06a19/64x,pad=8,sw=2 06a19.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/06a00/06a19/64x,pad=12,sw=2 06a19.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/06a00/06a19/64x,pad=16,sw=2 06a19.png"),

            # KVG 遠
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=4,sw=2 09060.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=8,sw=2 09060.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=12,sw=2 09060.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=16,sw=2 09060.png"),

            # KVG 園
            pathstr("~/datadisk/dataset/kanjivg/output/main/05700/05712/64x,pad=4,sw=2 05712.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05700/05712/64x,pad=8,sw=2 05712.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05700/05712/64x,pad=12,sw=2 05712.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05700/05712/64x,pad=16,sw=2 05712.png"),

            # KVG 亻 (倹)
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=4,sw=2 05039-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=8,sw=2 05039-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=12,sw=2 05039-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=16,sw=2 05039-g1.png"),
            
            # KVG 宀 (宇)
            pathstr("~/datadisk/dataset/kanjivg/output/main/05b00/05b87/64x,pad=4,sw=2 05b87-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05b00/05b87/64x,pad=8,sw=2 05b87-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05b00/05b87/64x,pad=12,sw=2 05b87-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05b00/05b87/64x,pad=16,sw=2 05b87-g1.png"),
            
            # KVG 广 (麻)
            pathstr("~/datadisk/dataset/kanjivg/output/main/09e00/09ebb/64x,pad=4,sw=2 09ebb-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09e00/09ebb/64x,pad=8,sw=2 09ebb-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09e00/09ebb/64x,pad=12,sw=2 09ebb-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09e00/09ebb/64x,pad=16,sw=2 09ebb-g1.png"),
            
            # KVG ⻌ (遠)
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=4,sw=2 09060-g8.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=8,sw=2 09060-g8.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=12,sw=2 09060-g8.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=16,sw=2 09060-g8.png"),

            # KVG 倹
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=4,sw=2 05039.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=8,sw=2 05039.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=12,sw=2 05039.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=16,sw=2 05039.png"),

            # KVG 困
            pathstr("~/datadisk/dataset/kanjivg/output/main/05600/056f0/64x,pad=4,sw=2 056f0.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05600/056f0/64x,pad=8,sw=2 056f0.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05600/056f0/64x,pad=12,sw=2 056f0.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05600/056f0/64x,pad=16,sw=2 056f0.png"),

            # KVG 麻
            pathstr("~/datadisk/dataset/kanjivg/output/main/09e00/09ebb/64x,pad=4,sw=2 09ebb.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09e00/09ebb/64x,pad=8,sw=2 09ebb.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09e00/09ebb/64x,pad=12,sw=2 09ebb.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09e00/09ebb/64x,pad=16,sw=2 09ebb.png"),
            
            # KVG 諭
            pathstr("~/datadisk/dataset/kanjivg/output/main/08a00/08aed/64x,pad=4,sw=2 08aed.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/08a00/08aed/64x,pad=8,sw=2 08aed.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/08a00/08aed/64x,pad=12,sw=2 08aed.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/08a00/08aed/64x,pad=16,sw=2 08aed.png"),
        ],

        device=torch.device(args.device),
    )
