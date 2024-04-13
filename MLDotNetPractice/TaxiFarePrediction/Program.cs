using System;
using System.IO;
using Microsoft.ML;
using TaxiFarePrediction.Model;

namespace TaxiFarePrediction
{
    /// <summary>
    /// ML.NET測試專案:
    /// - 主題: 透過回歸分析預測價格
    /// - 前置作業: Data資料夾必須放置要預測的資料集
    /// - 框架: 至少為NET 6.0
    /// - 概念: 提供Feature預測出Label
    /// </summary>
    internal class Program
    {
        private static string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        private static string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");

        static void Main(string[] args)
        {
            // 讀取資料來源

            // 初始化ML類別
            MLContext mlContext = new MLContext(seed: 0);

            // 訓練
            var model = Train(mlContext, _trainDataPath);

            // 評估
            Evaluate(mlContext, model);

            // 丟資料進去看看預測結果
            TestSinglePrediction(mlContext, model);
        }

        /// <summary>
        /// 訓練模型
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="dataPath"></param>
        /// <returns></returns>
        private static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // 透過ML.NET讀取資料源
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                // 演算法處理時使用的資料必須都是數值，須將內容為非數值的部分轉換為數值(另給名稱)
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                // 將所有和預測有關的欄位都放進去(參數1.為輸出的欄位名稱)
                // 備註: "旅程時間"考量每次車況都有所不同，不適合當作輸入預測
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                // 選擇要用的演算法(這裡用FastTree)
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            return model;
        }

        /// <summary>
        /// 透過訓練好的模型計算結果
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

            // 將test資料轉換為迴歸分析處理時需要的資料
            var predictions = model.Transform(dataView);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        /// <summary>
        /// 測試資料用
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
