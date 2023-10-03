# RadicalStylist

submodule として stable-diffusion を利用  
`-` を含むディレクトリは `import` できない（Syntax Error になる）ので `_` に置き換えている

## 部首データの準備

次の `Json` を用意して `--radical_data_path` に与える

```typescript
type Item = {
  // 部首名
  element: string;
  // その部首を構成する何番目のストロークか
  part: number | null;
  // 大きさ 0 ~ 1
  boundings: {
    left: number;
    right: number
    top: number;
    bottom: number;
  }
  children: Item[]; // 部首をさらに分解したもの
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
python sampling
```
