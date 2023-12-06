import argparse
from typing import Iterable

import torch

import character_utility as charutil
from character_decomposer import IdentityDecomposer
from dataset import DatasetProvider, EtlcdbDatasetProvider, RSDataset
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

    print(f"test images:", "\n\t".join(test_image_paths), sep="")

    vae = StableDiffusionVae(original_vae_path).to(device=device)

    print("training started!")
    vae.train(save_path, dataloader, epochs, test_image_paths)
    print("training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    main(
        # save_path=pathstr("./output/vae/ETL4(bw)"),
        save_path=pathstr("./output/vae/ETL9B"),
        original_vae_path=pathstr("~/datadisk/stable-diffusion-v1-5/vae"),

        image_size=64,

        batch_size=64,
        epochs=10000,
        shuffle_dataset=True,
        dataloader_num_workers=4,

        # datasets=[
        #     # 6120 件
        #     EtlcdbDatasetProvider(
        #         etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),
        #         etlcdb_process_type="black_and_white 64x64",
        #         etlcdb_name="ETL4",
        #         charnames=(charutil.all_kanas + charutil.kanjis.all()),
        #     ),
        # ],
        datasets=[
            # 610236 件
            EtlcdbDatasetProvider(
                etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),
                etlcdb_process_type="64x64",
                etlcdb_name="ETL9B",
                charnames=(charutil.all_kanas + charutil.kanjis.all()),
            ),
        ],

        test_image_paths=[
            # あ
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL4/5001/000000.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL4/5002/000051.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL4/5001/000000.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL4/5002/000051.png"),

            # ぬ
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL4/5003/000124.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL4/5004/000175.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL4/5003/000124.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL4/5004/000175.png"),

            # 何
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/1/000007.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/11/000963.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/1/000007.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/11/000963.png"),

            # 標
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/7/000653.png"),
            pathstr("~/datadisk/dataset/etlcdb/black_and_white 64x64/ETL8G/17/001609.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/7/000653.png"),
            pathstr("~/datadisk/dataset/etlcdb/no_background 64x64/ETL8G/17/001609.png"),

            # 何
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/2/607489.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/22/003325.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/42/006361.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/62/009397.png"),

            # 標
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/17/609644.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/37/005480.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/57/008516.png"),
            pathstr("~/datadisk/dataset/etlcdb/64x64/ETL9B/77/011552.png"),

            # 倹
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=4,sw=2 05039.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=8,sw=2 05039.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=12,sw=2 05039.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=16,sw=2 05039.png"),
            
            # 遠
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=4,sw=2 09060.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=8,sw=2 09060.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=12,sw=2 09060.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=16,sw=2 09060.png"),

            # 亻
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=4,sw=2 05039-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=8,sw=2 05039-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=12,sw=2 05039-g1.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/05000/05039/64x,pad=16,sw=2 05039-g1.png"),
            
            # ⻌
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=4,sw=2 09060-g8.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=8,sw=2 09060-g8.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=12,sw=2 09060-g8.png"),
            pathstr("~/datadisk/dataset/kanjivg/output/main/09000/09060/64x,pad=16,sw=2 09060-g8.png"),
        ],

        device=torch.device(args.device),
    )
