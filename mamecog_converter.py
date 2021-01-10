#
# mamecog_converter.py
# Copyright © 2020 Hideki Hashimoto
#
# https://github.com/84moto/mamecog
#
# This software is released under the MIT License.
#

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Kerasで作成された学習済みモデル（h5ファイル）を読み込み、
# Conv2D層とDense層のKernelとBiasを取り出して、
# まめコグC#ライブラリで読み込み可能なバイナリファイルに保存する。
# 
# h5ファイルは下記のように保存する。
#  ↓
# model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
# model.save("vgg16_model.h5")
# 
def mamecog_convert(src_h5_fname):

    model = keras.models.load_model(src_h5_fname)
    model.summary()

    layer_num = len(model.layers)
    for layer_idx in range(layer_num):
        print("\nLayer", layer_idx+1)
        layer = model.layers[layer_idx]
        print(layer.name)

        if isinstance(layer, keras.layers.Conv2D):
            print("Kernel shape")
            print(layer.kernel.shape)
            print("Bias shape")
            print(layer.bias.shape)

            # Kernelを下記のかたちのC#のfloat配列として読み込めるように
            # バイナリファイルに保存する
            # [出力面の数、入力面の数、カーネル縦サイズ、カーネル横サイズ]
            fname_k = layer.name + "_k.bin"
            save_bin_k = layer.kernel.numpy().transpose(3,2,0,1)
            save_bin_k.tofile(fname_k)
            print("=> " + fname_k)

            # Biasを[出力面の数]のサイズのfloat配列として
            # C#で読み込めるようにバイナリファイルに保存する
            fname_b = layer.name + "_b.bin"
            save_bin_b = layer.bias.numpy()
            save_bin_b.tofile(fname_b)
            print("=> " + fname_b)

        if isinstance(layer, keras.layers.Dense):
            print("Kernel shape")
            print(layer.kernel.shape)
            print("Bias shape")
            print(layer.bias.shape)

            # Kernelを[出力Cellの数、入力Cellの数]のfloat配列として
            # C#で読み込めるようにバイナリファイルに保存する
            fname_k = layer.name + "_k.bin"
            save_bin_k = layer.kernel.numpy().transpose(1,0)
            save_bin_k.tofile(fname_k)
            print("=> " + fname_k)

            # Biasを[出力面の数]のサイズのfloat配列として
            # C#で読み込めるようにバイナリファイルに保存する
            fname_b = layer.name + "_b.bin"
            save_bin_b = layer.bias.numpy()
            save_bin_b.tofile(fname_b)
            print("=> " + fname_b)

    print("\nConvert done.")

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2:
        mamecog_convert(args[1])
    else:
        print("Usage: mamecog_converter.py h5filename")

#end
