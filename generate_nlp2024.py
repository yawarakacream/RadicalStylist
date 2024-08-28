import os
from argparse import ArgumentParser

from tqdm import tqdm

import torch

import character_utility as charutil
from character_decomposer import ClusteringLabelDecomposer, IdentityDecomposer
from dataset import CharacterDecomposer
from radical_stylist import RadicalStylist
from train import prepare_radicallists_with_name
from utility import pathstr, char2code, save_images, save_single_image


otehon_text_1 = "　わたしは、なつやすみに、はじめて　かぶとむしをつかまえました。はやおきを　してうらやまにいきました。てで、つかまえるとき、どきどきしました。三びきも つかまえました。　らいねんは、くわがたむしもつかまえたいと　おもいました。"
otehon_text_2 = ("　直線の数などに目をつけると、形をなかまに分けることができます。　　　　　　　　" +
          "　３本の直線でかこまれた形を、三角形といいます。４本の直線でかこまれた形を、四角形といいます。　　　　　　　　　　　　　" +
          "　三角形や四角形のまわりの直線をへん、かどの点をちょう点といいます。かどがみんな直角になっている四角形を、長方形といいます。長方形のむかい合っているへんの長さは同じです。")
otehon_text_3 = "　日本のまわりの海には、たくさんのしゅるいのさかなが、すんでいます。つめたい海がすきなさかながいます。あたたかい海がすきなさかなもいます。わたり鳥のように、きせつごとに北から南へ、南から北へと、いどうしながらくらしているさかなもいます。　　　海岸ちかくのいそにすみついているさかなや、そこのほうにいて、あまりうごきまわらないさかなもいます。かたちも、色も、もようも、じつにさまざまです。ひろい海の、どこかで、どんなふうにくらしているのか、のぞいてみましょう。"
otehon_text_4 = "　日本のまわりの海には、たくさんのしゅるいの魚が、すんでいます。冷たい海がすきな魚がいます。あたたかい海がすきな魚もいます。わたり鳥のように、季節ごとに北から南へ、南から北へと、いどうしながらくらしている魚もいます。　　　　　　　　　　　　　海岸近くのいそにすみついている魚や、底のほうにいて、あまり動きまわらない魚もいます。形も、色も、もようも、じつにさまざまです。ひろい海の、どこかで、どんなふうにくらしているのか、のぞいてみましょう。"
otehon_text_5 = "　雑木林は、燃料にするまきや炭を作るために植えられた人工の林で、落ち葉も肥料として利用されました。まきや炭をほとんどつかわなくなった今では、雑木林のもつ役わりも少なくなってきたようにみえますが、雑木林にはたくさんの生きものがすみ、命が生まれています。コナラの木１本をみても、どれだけ多くの虫がコナラの葉や実をたべていきているかがわかります。木がかれても、べつの虫がきて、そこにまた命が生まれます。雑木林は、四季をとおして楽しい観察ができます。"
otehon_text_6 = "　自由自在に空中を飛び交っている虫たちと川のなかの魚とは、一見なんの関係もないようにみえます。しかし、谷川でつったイワナの胃の中を調べてみると、虫の幼虫が、たくさんみつかります。このことから、自然のなかでは、虫の幼虫は、魚たちにとって大切なエサであることがわかります。同時に、水面からは見えにくい水中の石のうらがわや、すきまに、意外に多くの幼虫が生息していることも想像できます。実際、流れのなかの小さな石ひとつにも、思いがけずたくさんの種類と数の虫がついていて、おどろかされます。"


otehon_chars = set(otehon_text_1 + otehon_text_2 + otehon_text_3 + otehon_text_4 + otehon_text_5 + otehon_text_6)
otehon_chars &= charutil.all_kanas | charutil.kanjis.all
otehon_kanas = otehon_chars & charutil.all_kanas
otehon_kanjis = otehon_chars & charutil.kanjis.all


def main(
    save_path: str,
    model_name: str,

    character_decomposer: CharacterDecomposer,
    generate_kana: bool,
    generate_kanji: bool,
    
    device: torch.device,
):
    print(f"save_path: {save_path}")
    print(f"device: {device}")

    print("loading RadicalStylist...")
    rs = RadicalStylist.load(save_path=save_path, model_name=model_name, device=device)
    # rs.vae.to(device=torch.device("cpu"))
    print("loaded.")

    save_directory = pathstr(rs.save_path, "generated", "nlp2024")

    charnames = []
    if generate_kana:
        charnames += sorted(otehon_kanas)
        assert False
    if generate_kanji:
        # 観類楽察種 は kodomo2023 に 1 枚もない
        charnames += sorted(otehon_kanjis)
    if not len(charnames):
        raise Exception()

    pbar = tqdm(charnames)
    for charname in pbar:
        charcode = char2code(charname)
        pbar.set_postfix(charname=charname, charcode=charcode)

        directory = pathstr(save_directory, charcode)
        os.makedirs(directory, exist_ok=False)

        radicallists_with_name = prepare_radicallists_with_name(character_decomposer, rs.radicalname2idx, [charname])
        radicallists = [r for _, r in radicallists_with_name]
        assert len(radicallists) == 1

        writers = ["kodomo2023"] * 500

        images_list = rs.sample(radicallists, writers)
        assert len(images_list) == 1
        images = images_list[0]

        save_images(images, pathstr(directory, f"{charcode}_sheet.png"), nrow=25)

        for i, image in enumerate(images):
            save_single_image(image, pathstr(directory, f"{charcode}_{i}.png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    test_mode = "radical768"
    print(f"{test_mode=}")
    
    if test_mode == "character384":
        save_path = pathstr("./output/nlp2024/character384 nlp2024(kana,kanji)")

        character_decomposer = IdentityDecomposer()

        generate_kana = False
        generate_kanji = True

    elif test_mode == "radical384":
        save_path = pathstr("./output/nlp2024/radical384 nlp2024(kanji)+KVG(pad={4,8,12,16},sw=2)")

        character_decomposer = ClusteringLabelDecomposer(
            kvg_path=pathstr("~/datadisk/dataset/kanjivg"),
            radical_clustering_name="edu+jis_l1,2 n_clusters=384 (imsize=16,sw=2,blur=2)",
        )

        generate_kana = False
        generate_kanji = True
        
    elif test_mode == "radical768":
        save_path = pathstr("./output/nlp2024/radical768 nlp2024(kanji)+KVG(pad={4,8,12,16},sw=2)")

        character_decomposer = ClusteringLabelDecomposer(
            kvg_path=pathstr("~/datadisk/dataset/kanjivg"),
            radical_clustering_name="edu+jis_l1,2 n_clusters=384 (imsize=16,sw=2,blur=2)",
        )

        generate_kana = False
        generate_kanji = True

    else:
        raise Exception()

    model_name = "checkpoint=1000"

    # chars 8 * writer 224: 約 30 分
    main(
        save_path=save_path,
        model_name=model_name,

        character_decomposer=character_decomposer,
        generate_kana=generate_kana,
        generate_kanji=generate_kanji,

        device=torch.device(args.device),
    )
