import argparse
import copy
from typing import Union

import torch

from dataset import RSDataset
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
    len_radicals_of_char: int,
    writer_mode: str,
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
    shuffle_radicallist_of_char: bool,
    radical_depth: str,
    
    corpuses: list[tuple],
    etlcdb_path: str,
    
    test_chars: list[Union[str, tuple[str, list[Radical]]]],
    test_writers: Union[list[str], int],
    
    device: torch.device,
):
    print(f"save_path: {save_path}")
    print(f"device: {device}")

    kvgcontainer = KvgContainer(kvg_path)

    # train data
    print(f"train data:")

    dataset = RSDataset(kvgcontainer, radical_depth=radical_depth, ignore_kana=True, writer_mode=writer_mode)

    for corpus in corpuses:
        corpus_type = corpus[0]

        if corpus_type == "kvg":
            _, mode, line_width = corpus

            if mode == "none":
                continue

            dataset.add_kvg(mode, image_size, line_width)

        elif corpus_type == "etlcdb":
            _, etlcdb_process_type, etlcdb_names = corpus
            etlcdb_names = etlcdb_names.split(",")

            for etlcdb_name in etlcdb_names:
                dataset.add_etlcdb(etlcdb_path, etlcdb_process_type, etlcdb_name)

        else:
            raise Exception(f"unknown corpus type: {corpus_type}")

    print(f"\tradicals: {len(dataset.radicalname2idx)}")
    print(f"\twriters: {dataset.writername2idx and len(dataset.writername2idx)}")
    print(f"\ttotal data size: {len(dataset)}")

    dataloader = dataset.create_dataloader(
        batch_size=batch_size,
        shuffle_dataset=shuffle_dataset,
        num_workers=dataloader_num_workers,
        shuffle_radicallist_of_char=shuffle_radicallist_of_char,
    )

    # test writer
    if writer_mode == "none":
        assert dataset.writername2idx is None
        assert isinstance(test_writers, int) and 0 < test_writers
        
    else:
        assert dataset.writername2idx is not None
        assert isinstance(test_writers, list)

        for w in test_writers:
            if w not in dataset.writername2idx:
                raise Exception("unknown test writer: {w}")
            
        print("test writers:", ", ".join(test_writers))

    # test chars
    print("test characters:")
    test_radicallists_with_name = prepare_radicallists_with_name(kvgcontainer, dataset.radicalname2idx, test_chars)
    for name, radicallist in test_radicallists_with_name:
        print(f"\t{name} = {' + '.join(map(lambda r: r.name, radicallist))}")

    vae = StableDiffusionVae(stable_diffusion_path, device)

    print("initializing RadicalStylist...")
    radical_stylist = RadicalStylist(
        save_path=save_path,

        radicalname2idx=dataset.radicalname2idx,
        writername2idx=dataset.writername2idx,

        vae=vae,

        image_size=image_size,
        dim_char_embedding=dim_char_embedding,
        len_radicals_of_char=len_radicals_of_char,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,

        learning_rate=learning_rate,
        ema_beta=ema_beta,
        diffusion_noise_steps=diffusion_noise_steps,
        diffusion_beta_start=diffusion_beta_start,
        diffusion_beta_end=diffusion_beta_end,

        device=device,
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
        save_path=pathstr("./output/test kvg(lw=3) writer_mode=dataset/ETL8G_400+KVG_radical (encode_type=3, radical_depth=max)"),
        stable_diffusion_path=pathstr("~/datadisk/stable-diffusion-v1-5"),
        kvg_path=pathstr("~/datadisk/dataset/kanjivg"),

        image_size=64,
        dim_char_embedding=768,
        len_radicals_of_char=12,
        writer_mode="dataset",
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
        shuffle_radicallist_of_char=True,
        radical_depth="max",

        corpuses=[
            ("etlcdb", "black_and_white 64x64", "ETL8G_400"),
            ("kvg", "radical", 3)
        ],
        etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),

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
        test_writers=[f"ETL8G_400" for _ in range(7)] + ["KVG_64x_lw=3"],

        device=torch.device(args.device),
    )
