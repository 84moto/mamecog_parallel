//
// SampleVGG.cs
// Copyright 2021 Hideki Hashimoto
//
// https://github.com/84moto/mamecog_parallel
//
// This software is released under the MIT License.
//

using System;
using System.Diagnostics;
using System.Drawing;

namespace Mamecog
{
    class SampleVGG
    {
        static void Main(string[] args)
        {
            // テスト用の入力データを用意する
            Console.WriteLine("テスト画像読み込み");
            LayerData2D input1 = new LayerData2D(3, 224, 224);
            string inputFilename = "test_input.png";    // 224x224ピクセルのRGB画像
            using (Bitmap inputImage = new Bitmap(Image.FromFile(inputFilename)))
            {
                Debug.Assert(inputImage.Height == 224);
                Debug.Assert(inputImage.Width == 224);
                for (int y = 0; y < inputImage.Height; y++)
                {
                    for (int x = 0; x < inputImage.Width; x++)
                    {
                        Color pixelData = inputImage.GetPixel(x, y);
                        float r = (float)pixelData.R - 123.68f;
                        float g = (float)pixelData.G - 116.779f;
                        float b = (float)pixelData.B - 103.939f;
                        input1.SetVal(0, y, x, b);     // BGR
                        input1.SetVal(1, y, x, g);
                        input1.SetVal(2, y, x, r);
                    }
                }
            }
            
            // Conv2DとDenseのインスタンスを生成する
            Conv2D block1Conv1 = new Conv2D(64, 3, 3, 3);
            Conv2D block1Conv2 = new Conv2D(64, 64, 3, 3);
            Conv2D block2Conv1 = new Conv2D(128, 64, 3, 3);
            Conv2D block2Conv2 = new Conv2D(128, 128, 3, 3);
            Conv2D block3Conv1 = new Conv2D(256, 128, 3, 3);
            Conv2D block3Conv2 = new Conv2D(256, 256, 3, 3);
            Conv2D block3Conv3 = new Conv2D(256, 256, 3, 3);
            Conv2D block4Conv1 = new Conv2D(512, 256, 3, 3);
            Conv2D block4Conv2 = new Conv2D(512, 512, 3, 3);
            Conv2D block4Conv3 = new Conv2D(512, 512, 3, 3);
            Conv2D block5Conv1 = new Conv2D(512, 512, 3, 3);
            Conv2D block5Conv2 = new Conv2D(512, 512, 3, 3);
            Conv2D block5Conv3 = new Conv2D(512, 512, 3, 3);
            Dense fc1 = new Dense(4096, 25088);
            Dense fc2 = new Dense(4096, 4096);
            Dense predictions = new Dense(1000, 4096);

            // Conv2DとDenseのカーネルとバイアスをファイルから読み込む
            Console.WriteLine("学習済みモデル読み込み");
            block1Conv1.LoadKernelAndBias("block1_conv1_k.bin", "block1_conv1_b.bin");
            block1Conv2.LoadKernelAndBias("block1_conv2_k.bin", "block1_conv2_b.bin");
            block2Conv1.LoadKernelAndBias("block2_conv1_k.bin", "block2_conv1_b.bin");
            block2Conv2.LoadKernelAndBias("block2_conv2_k.bin", "block2_conv2_b.bin");
            block3Conv1.LoadKernelAndBias("block3_conv1_k.bin", "block3_conv1_b.bin");
            block3Conv2.LoadKernelAndBias("block3_conv2_k.bin", "block3_conv2_b.bin");
            block3Conv3.LoadKernelAndBias("block3_conv3_k.bin", "block3_conv3_b.bin");
            block4Conv1.LoadKernelAndBias("block4_conv1_k.bin", "block4_conv1_b.bin");
            block4Conv2.LoadKernelAndBias("block4_conv2_k.bin", "block4_conv2_b.bin");
            block4Conv3.LoadKernelAndBias("block4_conv3_k.bin", "block4_conv3_b.bin");
            block5Conv1.LoadKernelAndBias("block5_conv1_k.bin", "block5_conv1_b.bin");
            block5Conv2.LoadKernelAndBias("block5_conv2_k.bin", "block5_conv2_b.bin");
            block5Conv3.LoadKernelAndBias("block5_conv3_k.bin", "block5_conv3_b.bin");
            fc1.LoadKernelAndBias("fc1_k.bin", "fc1_b.bin");
            fc2.LoadKernelAndBias("fc2_k.bin", "fc2_b.bin");
            predictions.LoadKernelAndBias("predictions_k.bin", "predictions_b.bin");

            // 各層の入出力の格納先を用意する
            LayerData2D block1Conv1Output = new LayerData2D(64, 224, 224);
            LayerData2D block1Conv2Output = new LayerData2D(64, 224, 224);
            LayerData2D block1PoolOutput = new LayerData2D(64, 112, 112);
            LayerData2D block2Conv1Output = new LayerData2D(128, 112, 112);
            LayerData2D block2Conv2Output = new LayerData2D(128, 112, 112);
            LayerData2D block2PoolOutput = new LayerData2D(128, 56, 56);
            LayerData2D block3Conv1Output = new LayerData2D(256, 56, 56);
            LayerData2D block3Conv2Output = new LayerData2D(256, 56, 56);
            LayerData2D block3Conv3Output = new LayerData2D(256, 56, 56);
            LayerData2D block3PoolOutput = new LayerData2D(256, 28, 28);
            LayerData2D block4Conv1Output = new LayerData2D(512, 28, 28);
            LayerData2D block4Conv2Output = new LayerData2D(512, 28, 28);
            LayerData2D block4Conv3Output = new LayerData2D(512, 28, 28);
            LayerData2D block4PoolOutput = new LayerData2D(512, 14, 14);
            LayerData2D block5Conv1Output = new LayerData2D(512, 14, 14);
            LayerData2D block5Conv2Output = new LayerData2D(512, 14, 14);
            LayerData2D block5Conv3Output = new LayerData2D(512, 14, 14);
            LayerData2D block5PoolOutput = new LayerData2D(512, 7, 7);
            float[] flattenOutput = new float[25088];
            float[] fc1Output = new float[4096];
            float[] fc2Output = new float[4096];
            float[] predictionsOutput = new float[1000];

            // 各層の出力を順に計算する
            Console.WriteLine("CNN実行開始");
            var sw = new System.Diagnostics.Stopwatch();
            sw.Start();
            Console.WriteLine("Block 1");
            block1Conv1.Conv(block1Conv1Output, input1, true);
            block1Conv1.ReLU(block1Conv1Output);
            block1Conv2.Conv(block1Conv2Output, block1Conv1Output, true);
            block1Conv2.ReLU(block1Conv2Output);
            MaxPool2D.Calc(block1PoolOutput, block1Conv2Output, 2);
            Console.WriteLine("Block 2");
            block2Conv1.Conv(block2Conv1Output, block1PoolOutput, true);
            block2Conv1.ReLU(block2Conv1Output);
            block2Conv2.Conv(block2Conv2Output, block2Conv1Output, true);
            block2Conv2.ReLU(block2Conv2Output);
            MaxPool2D.Calc(block2PoolOutput, block2Conv2Output, 2);
            Console.WriteLine("Block 3");
            block3Conv1.Conv(block3Conv1Output, block2PoolOutput, true);
            block3Conv1.ReLU(block3Conv1Output);
            block3Conv2.Conv(block3Conv2Output, block3Conv1Output, true);
            block3Conv2.ReLU(block3Conv2Output);
            block3Conv3.Conv(block3Conv3Output, block3Conv2Output, true);
            block3Conv3.ReLU(block3Conv3Output);
            MaxPool2D.Calc(block3PoolOutput, block3Conv3Output, 2);
            Console.WriteLine("Block 4");
            block4Conv1.Conv(block4Conv1Output, block3PoolOutput, true);
            block4Conv1.ReLU(block4Conv1Output);
            block4Conv2.Conv(block4Conv2Output, block4Conv1Output, true);
            block4Conv2.ReLU(block4Conv2Output);
            block4Conv3.Conv(block4Conv3Output, block4Conv2Output, true);
            block4Conv3.ReLU(block4Conv3Output);
            MaxPool2D.Calc(block4PoolOutput, block4Conv3Output, 2);
            Console.WriteLine("Block 5");
            block5Conv1.Conv(block5Conv1Output, block4PoolOutput, true);
            block5Conv1.ReLU(block5Conv1Output);
            block5Conv2.Conv(block5Conv2Output, block5Conv1Output, true);
            block5Conv2.ReLU(block5Conv2Output);
            block5Conv3.Conv(block5Conv3Output, block5Conv2Output, true);
            block5Conv3.ReLU(block5Conv3Output);
            MaxPool2D.Calc(block5PoolOutput, block5Conv3Output, 2);
            Console.WriteLine("FC");
            block5PoolOutput.Flatten(flattenOutput);
            fc1.Calc(fc1Output, flattenOutput);
            fc1.ReLU(fc1Output);
            fc2.Calc(fc2Output, fc1Output);
            fc2.ReLU(fc2Output);
            predictions.Calc(predictionsOutput, fc2Output);
            predictions.Softmax(predictionsOutput);
            sw.Stop();
            TimeSpan ts = sw.Elapsed;
            Console.WriteLine("CNN実行完了：実行時間 = {0}", ts);

            // VGG16モデルの学習済み1000カテゴリのうち先頭10カテゴリ分の確率を出力
            for(int i = 0; i < 10; i++)
            {
                Console.WriteLine("カテゴリ[{0}] = {1}", i, predictionsOutput[i]);
            }
        }
    }
}
