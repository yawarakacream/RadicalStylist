import json
import os
import argparse

from torch.utils.data import DataLoader

import torchvision

from radical_stylist import RadicalStylist
from dataset import Char, Radical, create_data_loader, EtlcdbDataset
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
    num_workers,
    
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
    
    # 謎
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # dataset
    datasets = []
    for corpus in corpuses:
        corpus_type = corpus[0]
        
        if corpus_type == "etlcdb":
            _, etlcdb_process_type, etlcdb_names = corpus
            datasets.append(EtlcdbDataset(etlcdb_path, etlcdb_process_type, etlcdb_names))
            
        else:
            raise Exception(f"unknown corpus type: {corpus_type}")
    
    # data loader
    data_loader, all_charnames, all_writers = create_data_loader(
        batch_size, True, num_workers,
        datasets, train_chars_filter, charname2radicaljson,
        transforms,
    )
    print(f"train data:")
    print(f"\tchars: {len(all_charnames)}")
    print(f"\twriters: {len(all_writers)}")
    print(f"\ttotal data size: {len(data_loader.dataset)}")
    
    # radicalname2idx
    radicalname2idx = {}
    for _, char, _ in data_loader.dataset:
        for radical in char.radicals:
            name = radical.name
            if name is None:
                continue
            
            if name in radicalname2idx:
                continue
                
            radicalname2idx[name] = len(radicalname2idx)

    print(f"radicals: {len(radicalname2idx)}")
    
    # writer2idx
    if learn_writer:
        writer2idx = {w: i for i, w in enumerate(all_writers)}
        
        assert type(test_writers) == list
        print("test writers:", ", ".join(test_writers))
        
    else:
        writer2idx = None
        
        assert type(test_writers) == int and 0 < test_writers

    print("test characters:")
    
    tmp = []
    for test_char in test_chars:
        if type(test_char) == str:
            tmp.append(Char.from_radicaljson(charname2radicaljson[test_char]))
        elif isinstance(test_char, Char):
            tmp.append(test_char)
        else:
            raise Exception()
        
        for r in tmp[-1].radicals:
            if r.name not in radicalname2idx:
                raise Exception(f"unsupported radical '{r.name}' was found in '{tmp[-1].name}'")
        
        print("\t" + tmp[-1].to_formula_string())
    
    test_chars = tmp
    del tmp
    
    print("initializing RadicalStylist")
    radical_stylist = RadicalStylist(
        save_path,
        stable_diffusion_path,
        
        radicalname2idx,
        writer2idx,
        
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
    radical_stylist.train(data_loader, epochs, test_chars, test_writers)
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
        save_path=pathstr("./datadisk/save_path ETL8G_400/normal"),
        stable_diffusion_path=pathstr("~/datadisk/stable-diffusion-v1-5"),
        radicals_data_path=pathstr("~/datadisk/radical_processed/KanjiVG/all.json"),
        
        image_size=64,
        dim_char_embedding=384,
        char_length=12,
        learn_writer=True,
        num_res_blocks=1,
        num_heads=4,
        
        learning_rate=0.0001,
        ema_beta=0.995,
        diffusion_noise_steps=1000,
        diffusion_beta_start=0.0001,
        diffusion_beta_end=0.02,
        
        diversity_lambda=0.0,
        
        batch_size=244,
        epochs=1000,
        num_workers=4,
        
        corpuses=[corpus("etlcdb/no_background 64x64/ETL8G_400")],
        train_chars_filter=None,
        etlcdb_path=pathstr("~/datadisk/etlcdb_processed"),
        
        test_chars=test_chars,
        test_writers=[f"ETL8G_400_{i}" for i in range(1, 8 + 1)],
        # test_writers=8,
        
        device=args.device,
    )
