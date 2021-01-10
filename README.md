# mamecog_parallel

マルチコア版まめコグ － C#用の小さなCNNエンジン

## 概要

マルチコア版まめコグ（mamecog_parallel）は、Pythonを使わずにC#だけでCNN（Convolutional Neural Network）の推論（分類・判定）を実行する小規模ライブラリです。
※ただし、CNNの学習にはPython（Keras）を用います。


まめコグは、C#による一般的なWindowsアプリケーションにCNNを組み込んで動作させることを目的としています。

まめコグでは、Kerasの学習済みモデルをC#で読み込み可能な独自形式に変換し、これを用いてC#アプリケーションでCNNの推論を実行します。
CNNモデルの学習はKerasとPythonで行いますが、学習完了後にC#アプリケーションでCNNによる認識・判別を実行するときにはKerasもPythonも不要です。

まめコグの最大の特徴は、他のC#ライブラリへの依存もなく、
小さな4個のC#クラス（LayerData2D, Conv2D, MaxPool2D, Dense）だけでCNNライブラリが構成されていることです。

マルチコア版まめコグ（mamecog_parallel）は、
無印まめコグ（[https://github.com/84moto/mamecog](https://github.com/84moto/mamecog)）をベースとして、
コンボリューション計算のループ処理を作り直し、Parallel.Forによるマルチコア処理を導入し高速化を実現しました。

## 使用方法

Keras公式ページの[Simple MNIST convnet（https://keras.io/examples/vision/mnist_convnet/）](https://keras.io/examples/vision/mnist_convnet/)をサンプルとして、まめコグC#ライブラリを用いてCNNを実行するための手順を説明します。

### 【手順１】Kerasで学習済みモデルを保存する

上記URLのサンプルでfit()メソッドの後に下記のコードを追加し、学習済みモデルを.h5ファイルとして保存します。

```
model.save("my_model.h5")
```

このとき、Kerasのsummary()メソッドで学習済みモデルの構造を確認しておきます。今回のMNISTサンプルでは次のようなサマリーとなります。

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                16010     
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________
```

### 【手順２】Kerasの学習済みモデルをまめコグ形式に変換する

まめコグでは、CNNを用いたC#アプリケーションを開発する前に、Kerasで構築したCNNの学習済みファイルを、まめコグC#ライブラリの各クラスで使用する独自形式のバイナリファイルに変換します。
手順１で作成したmy_model.h5ファイルを入力として、下記のように変換ツール（mamecog_converter.py）を実行します。

```
python mamecog_converter.py my_model.h
```

MNISTサンプルでは、mamecog_converter.pyは下記の6個のbinファイルを出力します。

```
conv2d_b.bin    conv2d_k.bin
conv2d_1_b.bin  conv2d_1_k.bin
dense_b.bin     dense_k.bin
```

これらのファイルには、手順１のKerasモデルのconv2d層、conv2d_1層、dense層の学習済みのカーネルとバイアスが、まめコグC#ライブラリ用に変換されて格納されています。

### 【手順３】C#で学習済みモデルを読み込む

次のようにConv2DとDenseのインスタンスを生成し、手順２のbinファイルを読み込みます。このとき、手順１で作成したCNNモデルの各層のサイズを指定します。

```
Conv2D conv1 = new Conv2D(32, 1, 3, 3);  //出力Ch数、入力Ch数、カーネル縦サイズ、カーネル横サイズ
Conv2D conv2 = new Conv2D(64, 32, 3, 3);
Dense dense = new Dense(10, 1600);       //出力Ch数、入力Ch数
conv1.LoadKernelAndBias("conv2d_k.bin", "conv2d_b.bin");
conv2.LoadKernelAndBias("conv2d_1_k.bin", "conv2d_1_b.bin");
dense.LoadKernelAndBias("dense_k.bin", "dense_b.bin");
```

### 【手順４】C#で各層のデータを保存する領域を用意する

Conv2DとMaxPool2Dの入出力を格納するためのLayerData2Dのインスタンスを作成します。Dense層の出力は1次元配列に格納します。

```
LayerData2D input0 = new LayerData2D(1, 28, 28);
LayerData2D conv1output = new LayerData2D(32, 26, 26);
LayerData2D pool1output = new LayerData2D(32, 13, 13);
LayerData2D conv2output = new LayerData2D(64, 11, 11);
LayerData2D pool2output = new LayerData2D(64, 5, 5);
float[] pool2flatten = new float[64 * 5 * 5];
float[] denseOutput = new float[10];
```

### 【手順５】C#でCNNの推論を実行する

下記のように各層の出力を順に計算します。

```
conv1.Conv(conv1output, input0, false);
conv1.ReLU(conv1output);
MaxPool2D.Calc(pool1output, conv1output, 2);
conv2.Conv(conv2output, pool1output, false);
conv2.ReLU(conv2output);
MaxPool2D.Calc(pool2output, conv2output, 2);
pool2output.Flatten(pool2flatten);
dense.Calc(denseOutput, pool2flatten);
dense.Softmax(denseOutput);
```

## VGG16の実行手順

まず、下記のPythonスクリプトを用いて、Kerasの学習済みVGG16モデルをダウンロードし.h5ファイルとして保存します。

```
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
model.save("vgg16_model.h5")
```

次に、前述のMNISTサンプルと同様に、mamecog_converter.pyを用いて.h5ファイルを変換して、まめコグ独自形式の重みバイナリファイルを生成します。

その後、SampleVGG.csのようにC#アプリケーションでの重みバイナリファイルの読み込みとVGG16のCNN推論を実行します。

## 動作環境および開発環境

下記の環境で開発と動作確認を行っています。
- Windows 10 Pro (64bit)
- Visual Studio Community 2019
- .Net Framework 4.7.2

CNNの学習済みモデルを下記の環境で構築しています。
- Python 3.7.8
- TensorFlow 2.2.0

## 参考

mamecog（まめコグ）とKerasの中間層の出力が同等であることを確認するためのツールとして、
mamecog_equality_check（[https://github.com/84moto/mamecog_equality_check](https://github.com/84moto/mamecog_equality_check)）を用意しました。
MNIST文字認識CNNでmamecogとKerasの中間層出力が同じになることを確認済みです。

## ライセンス

マルチコア版まめコグ（mamecog_parallel）は、MITライセンスで公開します。
"as is"（現状のまま）の提供です。一切の保証はありません。
ご使用は自己責任でお願いします。

## 開発者

マルチコア版まめコグ（mamecog_parallel）の開発者は、Hideki Hashimoto（84moto）です。
ご連絡は[https://twitter.com/hashimov](https://twitter.com/hashimov)にお願いします。

