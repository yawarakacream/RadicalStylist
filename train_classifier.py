import argparse

import torch

from classifier import RSClassifier
from dataset import RSDataset, create_dataloader
from image_vae import StableDiffusionVae
from utility import pathstr, create_charname2radicaljson


def main(
    save_path: str,
    stable_diffusion_path: str,
    radicals_data_path: str,
    
    image_size: int,
    grayscale: bool,
    
    learning_rate: float,
    
    batch_size: int,
    epochs: int,
    shuffle_dataset: bool,
    dataloader_num_workers: int,
    shuffle_radicals_of_char: bool,

    corpuses: list[str],
    etlcdb_path: str,
    
    device: torch.device,
):
    print(f"save_path: {save_path}")
    
    charname2radicaljson = create_charname2radicaljson(radicals_data_path)
    
    # dataset
    dataset = RSDataset(charname2radicaljson, ignore_kana=True)
    dataset.add_from_corpuses_string(corpuses, etlcdb_path)
    
    radicalname2idx = dataset.create_radicalname2idx()
    
    train_dataset, valid_dataset = dataset.random_split((0.9, 0.1))
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle_dataset=shuffle_dataset,
        num_workers=dataloader_num_workers,
        shuffle_radicals_of_char=shuffle_radicals_of_char,
        radicalname2idx=radicalname2idx,
        writername2idx=None,
    )
    
    valid_dataloader = create_dataloader(
        valid_dataset,
        batch_size=batch_size,
        shuffle_dataset=False,
        num_workers=dataloader_num_workers,
        shuffle_radicals_of_char=False,
        radicalname2idx=radicalname2idx,
        writername2idx=None,
    )
    
    print(f"train data:")
    print(f"\ttotal: {len(train_dataset)}")
    print(f"valid data:")
    print(f"\ttotal: {len(valid_dataset)}")
    
    # radicalname2idx
    radicalname2idx = dataset.create_radicalname2idx()
    print(f"radicals: {len(radicalname2idx)}")
    
    # 部首の出現数
    radicalcount = [0 for _ in range(len(radicalname2idx))]
    for _, chars, _ in train_dataloader:
        for c in chars:
            for r in c.radicals:
                radicalcount[r.idx] += 1
    
    bce_pos_weight = [(len(train_dataset) - rc) / rc for rc in radicalcount]
    
    vae = StableDiffusionVae(stable_diffusion_path, device)

    print("initializing RSClassifier")
    cf = RSClassifier(
        save_path=save_path,
        
        radicalname2idx=radicalname2idx,

        vae=vae,
        
        image_size=image_size,
        grayscale=grayscale,
        
        learning_rate=learning_rate,
        bce_pos_weight=bce_pos_weight,
        
        device=device,
    )
    cf.save(exist_ok=False)
    print("initialized")
    
    print("training started")
    cf.train(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,

        epochs=epochs,
    )
    print("training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()
    
    main(
        save_path=pathstr("./output/classifier test"),
        stable_diffusion_path=pathstr("~/datadisk/stable-diffusion-v1-5"),
        radicals_data_path=pathstr("~/datadisk/dataset/kanjivg/build/all.json"),
        
        image_size=64,
        grayscale=True,
        
        learning_rate=0.0001,
        
        batch_size=64,
        epochs=1000,
        shuffle_dataset=True,
        dataloader_num_workers=4,
        shuffle_radicals_of_char=True,
        
        corpuses=["etlcdb/no_background 64x64/ETL8G_400"],
        etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),
        
        device=torch.device(args.device),
    )
