# 画像スタイル変換ライブラリ

This is a fast neural style transfer implement with Keras 2(Tensorflow backend).
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

## Requirements

```bash
pip install -r requirements.txt
```

## Installation
Download pretrained style weight files.

```bash
cd conf/
bash download.sh

```

## directories

```bash
 ├─ conf          suffix が_weights.h5 となるスタイル画像の重みファイル
 │                スタイル変換時に参照される
 ├─ src           python ファイル格納場所
 ├─ docs          python APIドキュメント出力先
 └─ images
     ├─ content   画像入力ファイル
     ├─ output    スタイル変換後のファイル出力場所
     │            訓練時の一時ファイル出力場所
     ├─ style     スタイル画像, スタイル変換時/訓練時に参照
     ├─ train     サブフォルダを作成し、その下に訓練用のスタイル画像配置
     └─ train.org 訓練用のスタイル画像のバック・アップ場所
```

スタイル変換の適用可能なスタイル名は conf 以下の重みファイル

```bash
 ├─ conf   [style name]_weights.h5
 └─ images
     └─  style   スタイル画像
```

のペアが必要となります。

## transfer
- Generate a style converted image
  (1)元画像 + (2)スタイル画像 -> (3) スタイル変換後の画像

### 関連ファイル

```bash
 ├─ conf             スタイル画像の重みファイル [style name]_weights.h5
 ├─ src 
 │   ├─ transfer.py        スタイル変換実行 python ファイル
 │   └─ sample_transfer.sh スタイル変換サンプルスクリプト
 └─ images
     ├─ content      画像入力ファイル格納場所 -> (1)
     ├─ output       スタイル変換後のファイル出力場所 -> (3)
     └─ style        スタイル画像 -> (2)
```

### スタイル変換サンプル

```bash
cd src
bash sample_run.sh
```

## training
- Training a new style
  (1)訓練用の画像群 + (2)スタイル画像 -> (3) スタイルの重みファイル

### 関連ファイル

```bash
 ├─ src 
 │   ├─ train.py        スタイル訓練実行 python ファイル
 │   └─ sample_train.sh スタイル訓練サンプルスクリプト
 └─ images
     ├─ output       スタイル画像の重みファイル [style name]_weights.h5 ->(3)
     ├─ style        スタイル画像 -> (2)
     ├─ train        サブフォルダを作成し、その下に訓練用のスタイル画像配置 ->(1)
     └─ train.org    訓練用のスタイル画像のバック・アップ場所
```

### スタイル訓練サンプル

```bash
cd src
bash sample_train.sh
```

作成出来た重みファイルを利用してスタイル変換する際には、[style name]_weights.h5
を conf 以下に配置してスタイル変換可能にする。

```bash
cp images/output/[style name]_weights.h5 conf/
```


