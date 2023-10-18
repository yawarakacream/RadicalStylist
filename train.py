import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

import torchvision

from character import Char, Radical
from dataset import RSDataset, create_dataloader
from radical_stylist import RadicalStylist
from utility import pathstr


def main(
    save_path,
    stable_diffusion_path,
    radicals_data_path,
    
    image_size,
    dim_char_embedding,
    char_length,
    learn_writer,
    num_res_blocks,
    num_heads,
    
    learning_rate,
    ema_beta,
    diffusion_noise_steps,
    diffusion_beta_start,
    diffusion_beta_end,
    
    diversity_lambda,
    
    batch_size,
    epochs,
    shuffle_dataset,
    dataloader_num_workers,
    shuffle_radicals_of_char,
    
    corpuses,
    train_chars_filter,
    etlcdb_path,
    
    test_chars,
    test_writers,
    
    device,
):
    print(f"save: {save_path}")
    
    # charname2radicaljson
    with open(radicals_data_path) as f:
        radicals_data = json.load(f)
    charname2radicaljson = {radical["name"]: radical for radical in radicals_data}
    
    # dataset
    dataset = RSDataset(charname2radicaljson, ignore_kana=True)
    for corpus in corpuses:
        corpus_type = corpus[0]
        
        if corpus_type == "etlcdb":
            _, etlcdb_process_type, etlcdb_names = corpus
            for etlcdb_name in etlcdb_names:
                dataset.add_etlcdb(etlcdb_path, etlcdb_process_type, etlcdb_name)
            
        else:
            raise Exception(f"unknown corpus type: {corpus_type}")
    
    # radicalname2idx
    radicalname2idx = {}
    for _, char, _ in dataset:
        for radical in char.radicals:
            name = radical.name
            if name is None:
                continue
            
            if name in radicalname2idx:
                continue
                
            radicalname2idx[name] = len(radicalname2idx)

    # writer
    if learn_writer:
        writername2idx = {w: i for i, w in enumerate(dataset.all_writers)}
    else:
        writername2idx = None
    
    dataloader = create_dataloader(
        dataset,
        batch_size,
        shuffle_dataset,
        dataloader_num_workers,
        shuffle_radicals_of_char,
        radicalname2idx,
        writername2idx,
    )
    
    print(f"train data:")
    print(f"\tchars: {len(dataset.all_charnames)}")
    print(f"\tradicals: {len(radicalname2idx)}")
    print(f"\twriters: {len(dataset.all_writernames)}")
    print(f"\ttotal data size: {len(dataset)}")
    
    # test writer
    if learn_writer:
        assert isinstance(test_writers, list)
        print("test writers:", ", ".join(test_writernames))
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
    
    print("initializing RadicalStylist")
    radical_stylist = RadicalStylist(
        save_path,
        stable_diffusion_path,
        
        radicalname2idx,
        writername2idx,
        
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
        
        diversity_lambda,
        
        device,
    )
    print("initialized")
        
    print("training started")
    radical_stylist.train(dataloader, epochs, test_chars, test_writers)
    print("training finished")


def corpus(data: str):
    data = data.split("/")
    
    corpus_type = data[0]
    
    if corpus_type == "etlcdb":
        if len(data) != 3:
            raise Exception(f"illegal etlcdb: {data}")
        
        etlcdb_process_type = data[1]
        etlcdb_names = data[2].split(",")
        return corpus_type, etlcdb_process_type, etlcdb_names
    
    else:
        raise Exception(f"unknown corpus type: {corpus_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    # 訓練データにある字
    test_chars = list("何標園遠")
    
    # 部首単体
    test_chars += [
        Char("亻", [Radical("亻", left=0.078125, right=0.3125, top=0.140625, bottom=0.875)]),
        Char("宀", [Radical("宀", left=0.140625, right=0.859375, top=0.078125, bottom=0.375)]),
        Char("广", [Radical("广", left=0.078125, right=0.78125, top=0.078125, bottom=0.84375)]),
        Char("⻌", [Radical("⻌", left=0.109375, right=0.8125, top=0.15625, bottom=0.828125)])
    ]
    
    # 訓練データにないが部首データにある字
    test_chars += list("倹困麻諭")
    
    # 適当に作った字
    # test_chars += []
    
    main(
        # save_path=pathstr("./datadisk/save_path ETL8G_400/ignore_writer epochs=2000"),
        save_path=pathstr("./datadisk/tmp"),
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
        
        diversity_lambda=0.0,
        
        batch_size=244,
        epochs=100,
        shuffle_dataset=True,
        dataloader_num_workers=4,
        shuffle_radicals_of_char=True,
        
        corpuses=[corpus("etlcdb/no_background 64x64/ETL8G_400")],
        train_chars_filter=None,
        etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),
        
        test_chars=test_chars,
        # test_writers=[f"ETL8G_400_{i}" for i in range(1, 8 + 1)],
        test_writers=8,
        
        device=torch.device(args.device),
    )
