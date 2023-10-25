import argparse
import json
import random

import torch

from dataset import RSDataset, create_dataloader
from utility import pathstr
from vae_with_classifier import VaeWithClassifier


def main(
    save_path,
    stable_diffusion_path,
    
    image_size,
    grayscale,
    
    learning_rate,
    vae_loss_kl_weight,
    vae_loss_lpips_weight,
    cf_loss_weight,
    
    batch_size,
    epochs,
    shuffle_dataset,
    dataloader_num_workers,
    shuffle_radicals_of_char,

    corpuses,
    etlcdb_path,
    
    device,
):
    print(f"{save_path=}")
    
    # charname2radicaljson
    with open(pathstr("~/datadisk/dataset/kanjivg/build/all.json")) as f:
        radicals_data = json.load(f)
    charname2radicaljson = {radical["name"]: radical for radical in radicals_data}
    
    # dataset
    dataset = RSDataset(charname2radicaljson, ignore_kana=True)
    dataset.add_from_corpuses_string(corpuses, etlcdb_path)
    
    # radicalname2idx
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
    
    cf_loss_bce_pos_weight = [(len(train_dataset) - rc) / rc for rc in radicalcount]
    
    print("initializing VaeWithClassifier")
    vwc = VaeWithClassifier.new(
        save_path=save_path,
        stable_diffusion_path=stable_diffusion_path,
        
        charname2radicaljson=charname2radicaljson,
        radicalname2idx=radicalname2idx,
        
        image_size=image_size,
        grayscale=grayscale,
        
        learning_rate=learning_rate,
        vae_loss_kl_weight=vae_loss_kl_weight,
        vae_loss_lpips_weight=vae_loss_lpips_weight,
        cf_loss_weight=cf_loss_weight,
        cf_loss_bce_pos_weight=cf_loss_bce_pos_weight,
        
        device=device,
    )
    print("initialized")
    
    print("training started")
    vwc.train(
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
        save_path=pathstr("./datadisk/vwc vae_train=only_MSE auto_cf_loss_bce_pos_weight ETL8G_400"),
        stable_diffusion_path=pathstr("~/datadisk/stable-diffusion-v1-5"),
        
        image_size=64,
        grayscale=True,
        
        learning_rate=0.0001,
        vae_loss_kl_weight=0.000001, # original: 0.000001
        vae_loss_lpips_weight=0, # original: 0.1
        cf_loss_weight=0.1,
        
        batch_size=64,
        epochs=100,
        shuffle_dataset=True,
        dataloader_num_workers=4,
        shuffle_radicals_of_char=True,
        
        corpuses=["etlcdb/no_background 64x64/ETL8G_400"],
        etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),
        
        device=torch.device(args.device),
    )
