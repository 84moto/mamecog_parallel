//
// Dense.cs
// Copyright 2021 Hideki Hashimoto
//
// https://github.com/84moto/mamecog_parallel
//
// This software is released under the MIT License.
//

// 現在のバージョンは、KerasのDenseクラスのオプションのうち、
// 下記を選択した場合のみをサポートする。
//  ↓
// use_bias=True,
// kernel_regularizer=None,
// bias_regularizer=None,
// activity_regularizer=None,
// kernel_constraint=None,
// bias_constraint=None,

using System;
using System.IO;
using System.Threading.Tasks;

namespace Mamecog
{
    /// <summary>
    /// Dense層の計算を行うクラス
    /// </summary>
    public class Dense
    {
        public int OutputCellNum { get; }
        public int InputCellNum { get; }
        public float[] Kernel;
        public float[] Bias;

        /// <summary>
        /// 入出力層のサイズを指定してDense計算用オブジェクトを作成する
        /// </summary>
        /// <param name="outputCellNum">出力層のサイズ</param>
        /// <param name="inputCellNum">入力層のサイズ</param>
        public Dense(int outputCellNum, int inputCellNum)
        {
            OutputCellNum = outputCellNum;
            InputCellNum = inputCellNum;
            Kernel = new float[outputCellNum * inputCellNum];
            Bias = new float[outputCellNum];
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

        /// <summary>
        /// （デバッグ用）カーネルとバイアスの値を表示する
        /// </summary>
        public void PrintKernelAndBias()
        {
            Console.WriteLine("Number of Output Cells = " + OutputCellNum.ToString());
            Console.WriteLine("Number of Input Cells = " + InputCellNum.ToString());
            for (int outputCellIdx = 0; outputCellIdx < OutputCellNum; outputCellIdx++)
            {
                Console.WriteLine("Kernel -> " + outputCellIdx.ToString());
                for (int inputCellIdx = 0; inputCellIdx < InputCellNum; inputCellIdx++)
                {
                    float w = Kernel[InputCellNum * outputCellIdx + inputCellIdx];
                    Console.Write(w.ToString() + ", ");
                }
                Console.Write("\n");
            }
            Console.WriteLine("Bias");
            for (int outputCellIdx = 0; outputCellIdx < OutputCellNum; outputCellIdx++)
            {
                float w = Bias[outputCellIdx];
                Console.Write(w.ToString() + ", ");
            }
            Console.Write("\n");
        }

        /// <summary>
        /// Dense層の重み付き総和を計算する
        /// </summary>
        /// <param name="outputCells">出力層のデータを格納する1次元float配列</param>
        /// <param name="inputCells">入力層のデータが格納された1次元float配列</param>
        public void Calc(float[] outputCells, float[] inputCells)
        {
            if (outputCells.Length != OutputCellNum)
                throw new Exception("出力層サイズ不整合");
            if (inputCells.Length != InputCellNum)
                throw new Exception("入力層サイズ不整合");

            //for (int outputCellIdx = 0; outputCellIdx < outputCells.Length; outputCellIdx++)
            Parallel.For(0, outputCells.Length, outputCellIdx =>
            {
                float sum = 0;
                int kernelStartIdx = InputCellNum * outputCellIdx;
                for (int inputCellIdx = 0; inputCellIdx < inputCells.Length; inputCellIdx++)
                {
                    sum += Kernel[kernelStartIdx + inputCellIdx] * inputCells[inputCellIdx];
                }
                outputCells[outputCellIdx] = sum + Bias[outputCellIdx];
            });
        }

        /// <summary>
        /// 活性化関数（ReLU）を適用する
        /// </summary>
        /// <param name="cells">適用対象のDense層出力を格納した配列</param>
        public void ReLU(float[] cells)
        {
            for (int i = 0; i < cells.Length; i++)
            {
                if (cells[i] < 0)
                    cells[i] = 0;
            }
        }

        /// <summary>
        /// 活性化関数（Softmax）を適用する
        /// </summary>
        /// <param name="cells">適用対象のDense層出力を格納した配列</param>
        public void Softmax(float[] cells)
        {
            float sum = 0;
            for (int i = 0; i < cells.Length; i++)
            {
                float exp = (float)Math.Exp(cells[i]);
                cells[i] = exp;
                sum += exp;
            }
            for (int i = 0; i < cells.Length; i++)
            {
                cells[i] /= sum;
            }
        }
    }
}
