# RadicalStylist

submodule として stable-diffusion を利用  
`-` を含むディレクトリは `import` できない（Syntax Error になる）ので `_` に置き換えている

## 部首データの準備

[KanjiVG](https://kanjivg.tagaini.net/index.html) を次のように整形して `--kvg_path` に与える

各文字に対して次の `Json` を用意する

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

## 実行

### 学習

```
python train.py
```

### サンプリング

```
python sample.py
```
