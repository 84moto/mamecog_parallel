//
// MaxPool2D.cs
// Copyright © 2020 Hideki Hashimoto
//
// https://github.com/84moto/mamecog
//
// This software is released under the MIT License.
//

// 現在のバージョンは、KerasのMaxPooling2Dクラスのオプションのうち、
// 下記を選択した場合のみをサポートする。
// （他のオプションは今後サポートを追加予定）
// strides=None
// padding="valid"
// data_format=None

using System;

namespace Mamecog
{
    /// <summary>
    /// プーリング層の計算を行うクラス
    /// </summary>
    public static class MaxPool2D
    {
        /// <summary>
        /// プーリング計算を行う
        /// </summary>
        /// <param name="outputLayer">出力層のデータを格納するLayerData2Dオブジェクト</param>
        /// <param name="inputLayer">入力層のデータが格納されたLayerData2Dオブジェクト</param>
        /// <param name="poolSize">プーリングサイズ</param>
        public static void Calc(LayerData2D outputLayer, LayerData2D inputLayer, int poolSize)
        {
            if (outputLayer.PlaneNum != inputLayer.PlaneNum)
                throw new Exception("Plane数不整合");
            if (outputLayer.PlaneWidth != inputLayer.PlaneWidth / poolSize)
                throw new Exception("Planeサイズ不整合"); 
            if (outputLayer.PlaneHeight != inputLayer.PlaneHeight / poolSize)
                throw new Exception("Planeサイズ不整合"); 

            for (int outputPlane = 0; outputPlane < outputLayer.PlaneNum; outputPlane++)
            {
                int outputPlaneStartIdx = outputLayer.PlaneHeight * outputLayer.PlaneWidth * outputPlane;
                int inputPlane = outputPlane;
                int inputPlaneStartIdx = inputLayer.PlaneHeight * inputLayer.PlaneWidth * inputPlane;
                for (int outputY = 0; outputY < outputLayer.PlaneHeight; outputY++)
                {
                    int outputRowStartIdx = outputPlaneStartIdx + outputLayer.PlaneWidth * outputY;
                    int inputY0 = outputY * poolSize;
                    for (int outputX = 0; outputX < outputLayer.PlaneWidth; outputX++)
                    {
                        int outputCellIdx = outputRowStartIdx + outputX;
                        int inputX0 = outputX * poolSize;
                        float maxVal = float.MinValue;
                        for (int inputY = inputY0; inputY < inputY0 + poolSize; inputY++)
                        {
                            int inputRowStartIdx = inputPlaneStartIdx + inputLayer.PlaneWidth * inputY;
                            for (int inputX = inputX0; inputX < inputX0 + poolSize; inputX++)
                            {
                                int inputCellIdx = inputRowStartIdx + inputX;
                                float inputVal = inputLayer.Cells[inputCellIdx];
                                if(inputVal > maxVal)
                                    maxVal = inputVal;
                            }
                        }
                        outputLayer.Cells[outputCellIdx] = maxVal;
                    }
                }
            }
        }
    }
}
