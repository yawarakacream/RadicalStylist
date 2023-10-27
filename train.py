import argparse
import json
from typing import Union

import torch

from character import Char, Radical
from dataset import RSDataset, create_dataloader
from image_vae import StableDiffusionVae
from radical_stylist import RadicalStylist
from utility import pathstr, create_charname2radicaljson


def main(
    save_path: str,
    stable_diffusion_path: str,
    radicals_data_path: str,
    
    image_size: int,
    dim_char_embedding: int,
    char_length: int,
    learn_writer: bool,
    num_res_blocks: int,
    num_heads: int,
    
    learning_rate: float,
    ema_beta: float,
    diffusion_noise_steps: int,
    diffusion_beta_start: float,
    diffusion_beta_end: float,
    
    batch_size: int,
    epochs: int,
    shuffle_dataset: bool,
    dataloader_num_workers: int,
    shuffle_radicals_of_char: bool,
    
    corpuses: list[str],
    etlcdb_path: str,
    
    test_chars: list[Union[str, Char]],
    test_writers: Union[list[str], int],
    
    device: torch.device,
):
    print(f"save_path: {save_path}")
    
    charname2radicaljson = create_charname2radicaljson(radicals_data_path)
    
    # dataset
    dataset = RSDataset(charname2radicaljson, ignore_kana=True)
    dataset.add_from_corpuses_string(corpuses, etlcdb_path)
    
    radicalname2idx = dataset.create_radicalname2idx()
    writername2idx = dataset.create_writername2idx() if learn_writer else None
    
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle_dataset=shuffle_dataset,
        num_workers=dataloader_num_workers,
        shuffle_radicals_of_char=shuffle_radicals_of_char,
        radicalname2idx=radicalname2idx,
        writername2idx=writername2idx,
    )
    
    print(f"train data:")
    print(f"\tchars: {len(dataset.all_charnames)}")
    print(f"\tradicals: {len(radicalname2idx)}")
    print(f"\twriters: {len(dataset.all_writernames)}")
    print(f"\ttotal data size: {len(dataset)}")
    
    # test writer
    if learn_writer:
        assert isinstance(test_writers, list)
        print("test writers:", ", ".join(test_writers))
    else:
        assert isinstance(test_writers, int) and 0 < test_writers

    # test chars
    print("test characters:")
    tmp = []
    for test_char in test_chars:
        if isinstance(test_char, str):
            tmp.append(Char.from_radicaljson(charname2radicaljson[test_char]))
        elif isinstance(test_char, Char):
            tmp.append(test_char)
        else:
            raise Exception()
        
        tmp[-1].register_radicalidx(radicalname2idx)
        
        print("\t" + tmp[-1].to_formula_string())
    
    test_chars = tmp
    del tmp

    vae = StableDiffusionVae(stable_diffusion_path, device)
    
    print("initializing RadicalStylist")
    radical_stylist = RadicalStylist(
        save_path,
        
        radicalname2idx,
        writername2idx,
        
        vae,

        image_size,
        dim_char_embedding,
        char_length,
        num_res_blocks,
        num_heads,
        
        learning_rate,
        ema_beta,
        diffusion_noise_steps,
        diffusion_beta_start,
        diffusion_beta_end,
        
        device,
    )
    radical_stylist.save(exist_ok=False)
    print("initialized")
        
    print("training started")
    radical_stylist.train(dataloader, epochs, test_chars, test_writers)
    print("training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    main(
        save_path=pathstr("./output/rs test"),
        stable_diffusion_path=pathstr("~/datadisk/stable-diffusion-v1-5"),
        radicals_data_path=pathstr("~/datadisk/dataset/kanjivg/build/all.json"),

        image_size=64,
        dim_char_embedding=384,
        char_length=12,
        learn_writer=False,
        num_res_blocks=1,
        num_heads=4,

        learning_rate=0.0001,
        ema_beta=0.995,
        diffusion_noise_steps=1000,
        diffusion_beta_start=0.0001,
        diffusion_beta_end=0.02,

        batch_size=244,
        epochs=1000,
        shuffle_dataset=True,
        dataloader_num_workers=4,
        shuffle_radicals_of_char=True,

        corpuses=["etlcdb/no_background 64x64/ETL8G_400"],
        etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),

        test_chars=[
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
        # test_writers=[f"ETL8G_400_{i}" for i in range(1, 8 + 1)],
        test_writers=8,

        device=torch.device(args.device),
    )
