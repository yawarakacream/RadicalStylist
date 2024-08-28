# RadicalStylist

潜在表現を通して文字画像を生成する拡散モデル．  
文字種を部品種の列として与える．

Nikolaidou+, WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models, ICDAR, 2023 [[arXiv]](https://arxiv.org/abs/2303.16576) [[GitHub]](https://github.com/koninik/WordStylist)  を改変．

## Quick Start

コマンドラインは整備していないので，各ファイルの `main` 関数の引数を変更する．  
デバイスのみ `--device` で指定できる．

**クローン**

```
git clone --recursive git@github.com:yawarakacream/RadicalStylist.git
```

**VAE のチューニング**

潜在表現の扱いには [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) の Variational Autoencoder（VAE）を利用している．

文字画像をうまく扱えるようチューニングする．

```
python train_vae.py
```

**訓練**

主な引数：

- モデルを保存するパス
- VAE のパス
- スタイルの条件付け（なし・データセット単位・各データに紐付けられている書き手 ID 単位）
- コンテンツの条件付け（文字単位・部品単位）

<details>
  <summary>
    部品データの作成
  </summary>

  [KanjiVG](https://kanjivg.tagaini.net/index.html) を次のように整形して `--kvg_path` に与える．  
  [kanjivg-extractor](https://github.com/yawarakacream/kanjivg-extractor) を利用するとよい．

  各文字に対して次の `Json` を用意する．

  `p`: 16 進数 5 桁文字コード，下 2 桁切り捨て
  `c`: 16 進数 5 桁文字コード

  `{p}/{c}/{c}.json`

  ```typescript
  type Item = {
    // KanjiVG の `id`
    kvgid: string;
    // 部首名 (KanjiVG の `kvg:element`)
    name: string;
    // その部首を構成する何番目のストロークか (KanjiVG の `kvg:part`)
    part: number | null;
    // 部首としての場所 (KanjiVG の `kvg:position`)
    position: string | null;
    // SVG の path の d
    svg: string[];
    // 部首をさらに分解したもの
    children: Item[];
  };

  type Json = Item[];
  ```
</details>

<details>
  <summary>
    ETLCDB の前処理
  </summary>

  (TODO)
</details>

<details>
  <summary>
    こどもの字データの前処理
  </summary>

  (TODO)
</details>

```
python train.py
```

**生成**

主な引数：

- モデルが保存されているパス
- チェックポイント
- スタイルの条件付け
- コンテンツの条件付け

```
python sample.py
```
