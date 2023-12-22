from argparse import ArgumentParser
from typing import Iterable, Union

import torch

import character_utility as charutil
from character_decomposer import BoundingBoxDecomposer, ClusteringLabelDecomposer
from dataset import CharacterDecomposer, DatasetProvider, EtlcdbDatasetProvider, Kodomo2023DatasetProvider, KvgCompositionDatasetProvider, KvgDatasetProvider, RSDataset, Radical, WriterMode
from image_vae import StableDiffusionVae
from radical_stylist import RadicalStylist
from utility import pathstr


def prepare_radicallists_with_name(
    decomposer: CharacterDecomposer,
    radicalname2idx,
    chars: list[Union[str, tuple[str, list[Radical]]]]
):
    radicallists_with_name: list[tuple[str, list[Radical]]] = []
    for char in chars:
        if isinstance(char, str):
            if char.startswith("kvg:"):
                name = char
                kvgid = char[len("kvg:"):]
                radicallist = decomposer.get_decomposition_by_kvgid(kvgid)
            else:
                name = char
                radicallist = decomposer.get_decomposition_by_charname(name)
        elif isinstance(char, tuple):
            name, radicallist = char
        else:
            raise Exception()
        
        radicallists_with_name.append((name, radicallist))

    # set idx
    for _, radicallist in radicallists_with_name:
        for r in radicallist:
            r.set_idx(radicalname2idx)

    return radicallists_with_name


def main(
    *,

    save_path: str,

    vae_path: str,
    
    image_size: int,
    dim_radical_embedding: int,
    len_radicals_of_char: int,
    radical_position: str,
    writer_mode: WriterMode,
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

    character_decomposer: CharacterDecomposer,
    datasets: Iterable[DatasetProvider],
    
    test_chars: list[Union[str, tuple[str, list[Radical]]]],
    test_writers: Union[list[str], int],
    
    device: torch.device,
):
    print(f"save_path: {save_path}")
    print(f"device: {device}")

    vae = StableDiffusionVae(vae_path)

    # train data
    print(f"train data:")

    train_dataset = RSDataset(
        decomposer=character_decomposer,
        writer_mode=writer_mode,
        image_size=image_size,
    )
    print("\tprovider:")
    for provider in datasets:
        print(f"\t\t{provider.__class__.__name__}: ", end="", flush=True)
        l = len(train_dataset)
        train_dataset.append_by_provider(provider)
        print(len(train_dataset) - l)

    print(f"\tradicals: {len(train_dataset.radicalname2idx)}")
    print(f"\twriters: {train_dataset.writername2idx and len(train_dataset.writername2idx)}")
    print(f"\ttotal data size: {len(train_dataset)}")

    train_dataloader = train_dataset.create_dataloader(
        batch_size=batch_size,
        shuffle=shuffle_dataset,
        num_workers=dataloader_num_workers,
        shuffle_radicallist_of_char=shuffle_radicallist_of_char,
    )

    # test writer
    if writer_mode == "none":
        assert train_dataset.writername2idx is None
        assert isinstance(test_writers, int) and 0 < test_writers
        
    else:
        assert train_dataset.writername2idx is not None
        assert isinstance(test_writers, list)

        unknowns = [w for w in test_writers if w not in train_dataset.writername2idx]
        if len(unknowns):
                raise Exception(f"unknown test writer:", ", ".join(unknowns))

        print("test writers:", ", ".join(test_writers))

    # test chars
    print("test characters:")
    test_radicallists_with_name = prepare_radicallists_with_name(character_decomposer, train_dataset.radicalname2idx, test_chars)
    for name, radicallist in test_radicallists_with_name:
        print(f"\t{name} = {' + '.join(map(lambda r: r.name, radicallist))}")

    print("initializing RadicalStylist...")
    radical_stylist = RadicalStylist(
        save_path=save_path,

        radicalname2idx=train_dataset.radicalname2idx,
        writername2idx=train_dataset.writername2idx,

        vae=vae,

        image_size=image_size,
        dim_radical_embedding=dim_radical_embedding,
        len_radicals_of_char=len_radicals_of_char,
        radical_position=radical_position,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,

        learning_rate=learning_rate,
        ema_beta=ema_beta,
        diffusion_noise_steps=diffusion_noise_steps,
        diffusion_beta_start=diffusion_beta_start,
        diffusion_beta_end=diffusion_beta_end,

        device=device,
    )
    radical_stylist.save(model_name="init", exist_ok=False)
    print("initialized.")
        
    print("training started!")
    radical_stylist.train(
        train_dataloader,
        epochs,
        test_radicallists_with_name,
        test_writers,
    )
    print("training finished!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    nlp2024_kana_datasets = [
        #
        EtlcdbDatasetProvider(
            etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),
            etlcdb_name="ETL8G_onlyhira_train",
            preprocess_type="black_and_white 64x64",
        ),
        #
        EtlcdbDatasetProvider(
            etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),
            etlcdb_name="ETL5_train",
            preprocess_type="black_and_white 64x64",
        ),
        #
        Kodomo2023DatasetProvider(
            kodomo2023_path=pathstr("~/datadisk/dataset/kodomo2023"),
            process_type="black_and_white 64x64",
            charnames=charutil.all_kanas,
        ),
    ]

    nlp2024_kanji_datasets = [
        # 133281 件
        EtlcdbDatasetProvider(
            etlcdb_path=pathstr("~/datadisk/dataset/etlcdb"),
            etlcdb_name="ETL9G_onlykanji_train(sheet=1-1000)",
            preprocess_type="black_and_white 64x64",
        ),
        # 3719 件
        Kodomo2023DatasetProvider(
            kodomo2023_path=pathstr("~/datadisk/dataset/kodomo2023"),
            process_type="black_and_white 64x64",
            charnames=charutil.kanjis.all,
        ),
    ]

    main(
        save_path=pathstr("./output/nlp2024/nlp2024(kanji)+KVG(pad={4,8,12,16},sw=2) radenc=cl"),

        # vae_path=pathstr("~/datadisk/stable-diffusion-v1-5/vae"),
        vae_path=pathstr("./output/vae/SanariFont001(n_items=262140)/vae_100"),

        image_size=64,
        dim_radical_embedding=768,
        len_radicals_of_char=8,
        radical_position="clustering-label",
        writer_mode="dataset",
        num_res_blocks=1,
        num_heads=4,

        learning_rate=0.0001,
        ema_beta=0.995,
        diffusion_noise_steps=1000,
        diffusion_beta_start=0.0001,
        diffusion_beta_end=0.02,

        batch_size=244,
        epochs=5000,
        shuffle_dataset=True,
        dataloader_num_workers=4,
        shuffle_radicallist_of_char=True,

        # character_decomposer=BoundingBoxDecomposer(
        #     kvg_path=pathstr("~/datadisk/dataset/kanjivg"),
        #     depth="max",
        #     image_size=64,
        #     padding=4,
        #     stroke_width=2,
        # ),
        character_decomposer=ClusteringLabelDecomposer(
            kvg_path=pathstr("~/datadisk/dataset/kanjivg"),
            radical_clustering_name="edu+jis_l1,2 n_clusters=384 (imsize=16,sw=2,blur=2)",
        ),
        datasets=[
            *nlp2024_kanji_datasets,
            *[
                # 9964 件
                KvgDatasetProvider(
                    kvg_path=pathstr("~/datadisk/dataset/kanjivg"),
                    charnames=charutil.kanjis.all,
                    mode="radical",
                    padding=p,
                    stroke_width=2,
                )
                for p in (4, 8, 12, 16)
            ],
        ],

        test_chars=[
            # ETL9G にある & kodomo2023 にある
            *"海虫魚自",

            # ETL9G にある & kodomo2023 にない
            *"何標園遠",

            # 部首単体
            "kvg:05039-g1", # 亻 (倹)
            "kvg:05b87-g1", # 宀 (宇)
            "kvg:09ebb-g1", # 广 (麻)
            "kvg:09060-g8", # ⻌ (遠)
        ],
        test_writers=[
            *["kodomo2023" for _ in range(8)],
            *["ETL9G_onlykanji_train(sheet=1-1000)" for _ in range(8)],
            *[f"KVG(pad={p},sw=2)" for p in (4, 8, 12, 16)],
        ],

        device=torch.device(args.device),
    )
