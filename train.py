import argparse
import copy
from typing import Union

import torch

import character_utility as charutil
from dataset import RSDataset, create_dataloader
from image_vae import StableDiffusionVae
from kanjivg import KvgContainer
from radical import Radical
from radical_stylist import RadicalStylist
from utility import pathstr


def prepare_radicallists_with_name(
    kvgcontainer: KvgContainer,
    radicalname2idx,
    chars: list[Union[str, tuple[str, list[Radical]]]]
):
    radicallists_with_name: list[tuple[str, list[Radical]]] = []
    for char in chars:
        if isinstance(char, str):
            name = char
            radicallist: list[Radical] = []

            # 最も浅い分解
            stack = [Radical.from_kvg(kvgcontainer.get_kvg(name), kvg_path=kvgcontainer.kvg_path)]
            while len(stack):
                r = stack.pop()
                if r.name in radicalname2idx:
                    radicallist.insert(0, r)
                    continue
                if len(r.children) == 0:
                    raise Exception(f"unsupported character: {name}")
                stack += r.children

            radicallists_with_name.append((name, radicallist))
            
            # # 1 段階分解
            # decomp_radicallist = []
            # for radical in radicallist:
            #     if len(radical.children):
            #         decomp_radicallist += radical.children
            #     else:
            #         decomp_radicallist.append(radical)
            # if (len(decomp_radicallist) != len(radicallist)) or any(
            #     map(lambda x: x[0].name != x[1].name, zip(decomp_radicallist, radicallist))
            # ):
            #     decomp_radicallist = copy.deepcopy(decomp_radicallist)
            #     radicallists_with_name.append((name, decomp_radicallist))

            # 最大まで分解
            def f(radical: Radical):
                ret = []
                for radical0 in radical.children:
                    ret0 = f(radical0)
                    if len(ret0) == 0:
                        ret.append(None)
                    else:
                        ret += ret0
                
                if (len(ret) == 0 or any(map(lambda r: r is None, ret))) and radical.name in radicalname2idx:
                    ret = [radical]
                return ret

            decomp_radicallist = []
            for r in radicallist:
                decomp_radicallist += f(r)
            assert len(decomp_radicallist) != 0
            if (len(decomp_radicallist) != len(radicallist)) or any(
                map(lambda x: x[0].name != x[1].name, zip(decomp_radicallist, radicallist))
            ):
                decomp_radicallist = copy.deepcopy(decomp_radicallist)
                radicallists_with_name.append((name, decomp_radicallist))
                
        elif isinstance(char, tuple):
            name, radicallist = char
            radicallists_with_name.append((name, radicallist))

        else:
            raise Exception()

    # set idx
    for _, radicallist in radicallists_with_name:
        for r in radicallist:
            r.set_idx(radicalname2idx)

    return radicallists_with_name

def main(
    *,

    save_path: str,
    stable_diffusion_path: str,
    kvg_path: str,
    
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
    radical_depth: str,
    
    corpuses: list[tuple],
    etlcdb_path: str,
    train_kvg: str,
    train_kvg_line_width: int,
    
    test_chars: list[Union[str, tuple[str, list[Radical]]]],
    test_writers: Union[list[str], int],
    
    device: torch.device,
):
    print(f"device: {device}")
    print(f"save_path: {save_path}")
    
    kvgcontainer = KvgContainer(kvg_path)
    
    # train data
    print(f"train data:")

    dataset = RSDataset(kvgcontainer, radical_depth=radical_depth, ignore_kana=True)

    for corpus in corpuses:
        corpus_type = corpus[0]

        if corpus_type == "etlcdb":
            _, etlcdb_process_type, etlcdb_names = corpus
            etlcdb_names = etlcdb_names.split(",")

            for etlcdb_name in etlcdb_names:
                dataset.add_etlcdb(etlcdb_path, etlcdb_process_type, etlcdb_name)

        else:
            raise Exception(f"unknown corpus type: {corpus_type}")

    if train_kvg == "none":
        pass
    
    elif train_kvg == "char":
        writername = f"KVG_{image_size}x_lw={train_kvg_line_width}"

        for charname in charutil.kanjis.all():
            kvg = kvgcontainer.get_kvg(charname)
            image_path = kvg.get_image_path(kvg_path, image_size, train_kvg_line_width)
            dataset.add_item(image_path, charname, writername)

    elif train_kvg == "radicals":
        writername = f"KVG_{image_size}x_lw={train_kvg_line_width}"

        for charname in charutil.kanjis.all():
            if charname in charutil.all_kanas:
                continue

            kvg = kvgcontainer.get_kvg(charname)

            stack = [kvg]
            while len(stack):
                kvg = stack.pop()
                stack += kvg.children

                if kvg.name is None:
                    continue

                radicalname = kvg.name # 雑
                if kvg.part is not None:
                    radicalname = f"{radicalname}_{kvg.part}"

                image_path = kvg.get_image_path(kvg_path, image_size, train_kvg_line_width)
                radical = Radical.from_kvg(kvg, kvg_path, image_path=image_path)
                dataset.add_item(image_path, radical, writername)

    else:
        raise Exception(f"unknown train_kvg: {train_kvg}")

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
    
    print(f"\tradicals: {len(radicalname2idx)}")
    print(f"\twriters: {writername2idx and len(writername2idx)}")
    print(f"\ttotal data size: {len(dataset)}")
    
    # test writer
    if learn_writer:
        assert isinstance(test_writers, list)
        print("test writers:", ", ".join(test_writers))
    else:
        assert isinstance(test_writers, int) and 0 < test_writers

    # test chars
    print("test characters:")
    test_radicallists_with_name = prepare_radicallists_with_name(kvgcontainer, radicalname2idx, test_chars)
    for name, radicallist in test_radicallists_with_name:
        print(f"\t{name} = {' + '.join(map(lambda r: r.name, radicallist))}")

    vae = StableDiffusionVae(stable_diffusion_path, device)

    print("initializing RadicalStylist...")
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
    print("initialized.")
        
    print("training started!")
    radical_stylist.train(dataloader, epochs, test_radicallists_with_name, test_writers)
    print("training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:2")
    args = parser.parse_args()
    
    main(
        # save_path=pathstr("./output/test kvg(lw=3) ignore_writer/ETL8G_400 (encode_type=3, radical_depth=max)"),
        # save_path=pathstr("./output/test kvg(lw=3) ignore_writer/ETL8G_400+KVG (encode_type=3, radical_depth=max)"),
        save_path=pathstr("./output/test kvg(lw=3) ignore_writer/ETL8G_400+KVG_radicals (encode_type=3, radical_depth=max)"),
        stable_diffusion_path=pathstr("~/datadisk/stable-diffusion-v1-5"),
        kvg_path=pathstr("~/datadisk/dataset/kanjivg"),

        image_size=64,
        dim_char_embedding=768,
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
        epochs=10000,
        shuffle_dataset=True,
        dataloader_num_workers=4,
        shuffle_radicals_of_char=True,
        radical_depth="max",

        corpuses=[("etlcdb", "no_background 64x64", "ETL8G_400")],
        etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),
        # train_kvg="none",
        # train_kvg="char",
        train_kvg="radicals",
        train_kvg_line_width=3,

        test_chars=[
            # ETL8G にある字
            *"何標園遠",

            # 部首単体
            ("亻", [Radical("亻", left=0.078125, right=0.3125, top=0.140625, bottom=0.875)]),
            ("宀", [Radical("宀", left=0.140625, right=0.859375, top=0.078125, bottom=0.375)]),
            ("广", [Radical("广", left=0.078125, right=0.78125, top=0.078125, bottom=0.84375)]),
            ("⻌", [Radical("⻌", left=0.109375, right=0.8125, top=0.15625, bottom=0.828125)]),

            # ETL8G にないが KVG にある字
            *"倹困麻諭",
        ],
        # test_writers=[f"ETL8G_400_{i}" for i in range(1, 8 + 1)],
        test_writers=8,

        device=torch.device(args.device),
    )
