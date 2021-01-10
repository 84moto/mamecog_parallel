//
// LayerData2D.cs
// Copyright © 2020 Hideki Hashimoto
//
// https://github.com/84moto/mamecog
//
// This software is released under the MIT License.
//

using System;
using System.IO;

namespace Mamecog
{
    /// <summary>
    /// CNNのひとつの層の出力値を保持するクラス
    /// </summary>
    public class LayerData2D
    {
        public int PlaneNum { get; }
        public int PlaneHeight { get; }
        public int PlaneWidth { get; }
        public float[] Cells;

        /// <summary>
        /// 面の数と面のサイズを指定して2D層のデータ保持オブジェクトを作成する
        /// </summary>
        /// <param name="planeNum">2D層に含まれる面の数</param>
        /// <param name="height">面のサイズ（縦）</param>
        /// <param name="width">面のサイズ（横）</param>
        public LayerData2D(int planeNum, int height, int width)
        {
            PlaneNum = planeNum;
            PlaneHeight = height;
            PlaneWidth = width;
            Cells = new float[planeNum * height * width];
        }

        public float GetVal(int plane, int y, int x)
        {
            return Cells[PlaneWidth * (PlaneHeight * plane + y) + x];
        }

        public void SetVal(int plane, int y, int x, float val)
        {
            Cells[PlaneWidth * (PlaneHeight * plane + y) + x] = val;
        }

        /// <summary>
        /// ファイルから2D層の値を読み込む
        /// </summary>
        public void LoadCellValues(string filename)
        {
            using (Stream stream = File.OpenRead(filename))
            {
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    for (int i = 0; i < Cells.Length; i++)
                    {
                        float f = reader.ReadSingle();
                        Cells[i] = f;
                    }
                }
            }
        }

        /// <summary>
        /// 2D層のデータをFlatten形式の並びに変換する
        /// </summary>
        /// <param name="flattenLayer">変換されたFlatten形式のデータを格納するfloat配列</param>
        public void Flatten(float[] flattenLayer)
        {
            if (flattenLayer.Length != Cells.Length)
                throw new Exception("Cell数不整合");

            int outputIdx = 0;
            for (int y = 0; y < PlaneHeight; y++)
            {
                for (int x = 0; x < PlaneWidth; x++)
                {
                    for (int inputPlane = 0; inputPlane < PlaneNum; inputPlane++)
                    {
                        float cellVal = GetVal(inputPlane, y, x);
                        flattenLayer[outputIdx] = cellVal;
                        outputIdx++;
                    }
                }
            }
        }

        /// <summary>
        /// （デバッグ用）2D層の出力の値を表示する
        /// </summary>
        public void PrintCellValues()
        {
            Console.WriteLine("Number of Planes = " + PlaneNum.ToString());
            Console.WriteLine("Plane Height = " + PlaneHeight.ToString());
            Console.WriteLine("Plane Width = " + PlaneWidth.ToString());
            for (int planeIdx = 0; planeIdx < PlaneNum; planeIdx++)
            {
                Console.WriteLine("Plane " + planeIdx.ToString());
                for (int y = 0; y < PlaneHeight; y++)
                {
                    for (int x = 0; x < PlaneWidth; x++)
                    {
                        float w = GetVal(planeIdx, y, x);
                        Console.Write(w.ToString("F4") + " ");
                    }
                    Console.Write("\n");
                }
            }
        }
    }
}
