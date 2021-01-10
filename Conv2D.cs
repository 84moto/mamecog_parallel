//
// Conv2D.cs
// Copyright 2021 Hideki Hashimoto
//
// https://github.com/84moto/mamecog_parallel
//
// This software is released under the MIT License.
//

// 現在のバージョンは、
// KerasのConv2Dクラスのオプションのうち、
// 下記を選択した場合のみをサポートする。
//  ↓
// kernel_sizeを奇数とする
// strides=(1, 1)
// data_format=None
// dilation_rate=(1, 1)
// groups=1
// use_bias=True
// kernel_regularizer=None
// bias_regularizer=None
// activity_regularizer=None
// kernel_constraint=None
// bias_constraint=None

using System;
using System.IO;
using System.Threading.Tasks;

namespace Mamecog
{
    /// <summary>
    /// Conv2D計算クラス
    /// </summary>
    public class Conv2D
    {
        public int OutputPlaneNum { get; }  // 出力層の面の数（チャンネル数）
        public int InputPlaneNum { get; }   // 入力層の面の数（チャンネル数）
        public int KernelHeight { get; }    // カーネスサイズ（縦）
        public int KernelWidth { get; }     // カーネスサイズ（横）
        public float[] Kernel;              // カーネル重み
        public float[] Bias;                // バイアス値

        /// <summary>
        /// カーネルの形状を指定してConv2D計算用オブジェクトを作成する
        /// </summary>
        /// <param name="outputPlaneNum">出力層の面の数</param>
        /// <param name="inputPlaneNum">入力層の面の数</param>
        /// <param name="kernelHeight">カーネスサイズ（縦）</param>
        /// <param name="kernelWidth">カーネスサイズ（横）</param>
        public Conv2D(int outputPlaneNum, int inputPlaneNum, int kernelHeight, int kernelWidth)
        {
            OutputPlaneNum = outputPlaneNum;
            InputPlaneNum = inputPlaneNum;
            KernelHeight = kernelHeight;
            KernelWidth = kernelWidth;
            Kernel = new float[outputPlaneNum * inputPlaneNum * kernelHeight * kernelWidth];
            Bias = new float[outputPlaneNum];
        }

        /// <summary>
        /// ファイルからカーネルとバイアスの値を読み込む
        /// </summary>
        /// <param name="kernelFileName">カーネルデータのファイル名</param>
        /// <param name="biasFileName">バイアスデータのファイル名</param>
        public void LoadKernelAndBias(string kernelFileName, string biasFileName)
        {
            using (Stream stream = File.OpenRead(kernelFileName))
            {
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    for (int i = 0; i < Kernel.Length; i++)
                    {
                        float f = reader.ReadSingle();
                        Kernel[i] = f;
                    }
                }
            }
            using (Stream stream = File.OpenRead(biasFileName))
            {
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    for (int i = 0; i < Bias.Length; i++)
                    {
                        float f = reader.ReadSingle();
                        Bias[i] = f;
                    }
                }
            }
        }

        public float GetKernelVal(int outputPlane, int inputPlane, int kernelY, int kernelX)
        {
            int idx = InputPlaneNum * KernelHeight * KernelWidth * outputPlane
                    + KernelHeight * KernelWidth * inputPlane
                    + KernelWidth * kernelY + kernelX;
            return Kernel[idx];
        }

        /// <summary>
        /// （デバッグ用）カーネルとバイアスの値を表示する
        /// </summary>
        public void PrintKernelAndBias()
        {
            Console.WriteLine("Number of Output Planes = " + OutputPlaneNum.ToString());
            Console.WriteLine("Number of Input Planes = " + InputPlaneNum.ToString());
            for (int outputPlane = 0; outputPlane < OutputPlaneNum; outputPlane++)
            {
                for (int inputPlane = 0; inputPlane < InputPlaneNum; inputPlane++)
                {
                    Console.WriteLine("Kernel " + inputPlane.ToString() + " -> " + outputPlane.ToString());
                    for (int kernelY = 0; kernelY < KernelHeight; kernelY++)
                    {
                        for (int kernelX = 0; kernelX < KernelWidth; kernelX++)
                        {
                            float w = GetKernelVal(outputPlane, inputPlane, kernelY, kernelX);
                            Console.Write(w.ToString() + ", ");
                        }
                        Console.Write("\n");
                    }
                }
            }
            Console.WriteLine("Bias");
            for (int outputPlane = 0; outputPlane < OutputPlaneNum; outputPlane++)
            {
                float w = Bias[outputPlane];
                Console.Write(w.ToString() + ", ");
            }
            Console.Write("\n");
        }

        /// <summary>
        /// 畳み込み計算を行う
        /// </summary>
        /// <param name="outputLayer">出力層のデータを格納するLayerData2Dオブジェクト</param>
        /// <param name="inputLayer">入力層のデータが格納されたLayerData2Dオブジェクト</param>
        /// <param name="withPadding">パディングありのときtrue、なしのときfalse</param>
        public void Conv(LayerData2D outputLayer, LayerData2D inputLayer, bool withPadding)
        {
            int outputPlaneWH = outputLayer.PlaneWidth * outputLayer.PlaneHeight;
            int inputPlaneWH = inputLayer.PlaneWidth * inputLayer.PlaneHeight;
            int kernelWH = KernelWidth * KernelHeight;
            int kernelHalfWidth = KernelWidth / 2;
            int kernelHalfHeight = KernelHeight / 2; 

            if (outputLayer.PlaneNum != OutputPlaneNum)
                throw new Exception("OutputPlaneNum不整合");
            if (inputLayer.PlaneNum != InputPlaneNum)
                throw new Exception("InputPlaneNum不整合");
            if (withPadding)
            {
                if (outputLayer.PlaneHeight != inputLayer.PlaneHeight)
                    throw new Exception("Planeサイズ不整合");
                if (outputLayer.PlaneWidth != inputLayer.PlaneWidth)
                    throw new Exception("Planeサイズ不整合");
            }
            else
            {
                if (outputLayer.PlaneHeight != inputLayer.PlaneHeight - kernelHalfHeight * 2)
                    throw new Exception("Planeサイズ不整合");
                if (outputLayer.PlaneWidth != inputLayer.PlaneWidth - kernelHalfWidth * 2)
                    throw new Exception("Planeサイズ不整合");
            }

            //カーネル（重み行列）のそれぞれの重みを入出力面の縦横のどの範囲に適用するか
            int[] outputStartX = new int[KernelWidth];
            int[] outputStartY = new int[KernelHeight];
            int[] inputStartX = new int[KernelWidth];
            int[] inputStartY = new int[KernelHeight];
            int[] loopLengthX = new int[KernelWidth];
            int[] loopLengthY = new int[KernelHeight];
            if (withPadding)
            {
                for (int kernelY = 0; kernelY < KernelHeight; kernelY++)
                {
                    outputStartY[kernelY] = Math.Max(kernelHalfHeight - kernelY, 0);
                    inputStartY[kernelY] = Math.Max(kernelY - kernelHalfHeight, 0);
                    loopLengthY[kernelY] = outputLayer.PlaneHeight - Math.Abs(kernelHalfHeight - kernelY);
                }
                for (int kernelX = 0; kernelX < KernelWidth; kernelX++)
                {
                    outputStartX[kernelX] = Math.Max(kernelHalfWidth - kernelX, 0);
                    inputStartX[kernelX] = Math.Max(kernelX - kernelHalfWidth, 0);
                    loopLengthX[kernelX] = outputLayer.PlaneWidth - Math.Abs(kernelHalfWidth - kernelX);
                }
            }
            else
            {
                for (int kernelY = 0; kernelY < KernelHeight; kernelY++)
                {
                    outputStartY[kernelY] = 0;
                    inputStartY[kernelY] = kernelY;
                    loopLengthY[kernelY] = outputLayer.PlaneHeight;
                }
                for (int kernelX = 0; kernelX < KernelWidth; kernelX++)
                {
                    outputStartX[kernelX] = 0;
                    inputStartX[kernelX] = kernelX;
                    loopLengthX[kernelX] = outputLayer.PlaneWidth;
                }
            }

            Array.Clear(outputLayer.Cells, 0, outputLayer.Cells.Length);

            //for (int outputPlane = 0; outputPlane < outputLayer.PlaneNum; outputPlane++)
            Parallel.For(0, outputLayer.PlaneNum, outputPlane =>
            {
                int outputPlaneStartIdx = outputPlaneWH * outputPlane;
                for (int inputPlane = 0; inputPlane < inputLayer.PlaneNum; inputPlane++)
                {
                    int inputPlaneStartIdx = inputPlaneWH * inputPlane;
                    int kernelStartIdx = kernelWH * (InputPlaneNum * outputPlane + inputPlane);
                    for (int kernelY = 0; kernelY < KernelHeight; kernelY++)
                    {
                        int kernelRowStartIdx = kernelStartIdx + KernelWidth * kernelY;
                        int outputY0 = outputStartY[kernelY];
                        int inputY0 = inputStartY[kernelY];
                        int loopLenY = loopLengthY[kernelY];
                        for (int kernelX = 0; kernelX < KernelWidth; kernelX++)
                        {
                            //float w = GetKernelVal(outputPlane, inputPlane, kernelY, kernelX);
                            float w = Kernel[kernelRowStartIdx + kernelX];
                            int outputX0 = outputStartX[kernelX];
                            int inputX0 = inputStartX[kernelX];
                            int loopLenX = loopLengthX[kernelX];
                            int outputY = outputY0;
                            int inputY = inputY0;
                            for (int y = 0; y < loopLenY; y++)
                            {
                                int outputRowStartIdx = outputPlaneStartIdx + outputLayer.PlaneWidth * outputY;
                                int inputRowStartIdx = inputPlaneStartIdx + inputLayer.PlaneWidth * inputY;
                                Span<float> outputSpan = new Span<float>(outputLayer.Cells, outputRowStartIdx + outputX0, loopLenX);
                                Span<float> inputSpan = new Span<float>(inputLayer.Cells, inputRowStartIdx + inputX0, loopLenX);
                                for (int x = 0; x < loopLenX; x++)
                                {
                                    outputSpan[x] += w * inputSpan[x];
                                }
                                outputY++;
                                inputY++;
                            }
                        }
                    }
                }
            });
        }

        /// <summary>
        /// 活性化関数（ReLU）を適用する
        /// </summary>
        /// <param name="layer">適用対象のLayerData2Dオブジェクト</param>
        public void ReLU(LayerData2D layer)
        {
            for (int i = 0; i < layer.Cells.Length; i++)
            {
                if (layer.Cells[i] < 0)
                    layer.Cells[i] = 0f;
            }
        }
    }
}
