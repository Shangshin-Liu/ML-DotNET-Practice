using Microsoft.ML.Data;

namespace TaxiFarePrediction.Model
{
    /// <summary>
    /// 定義要輸入的計程車旅程資料    
    /// </summary>
    public class TaxiTrip
    {
        /// <summary>
        /// 供應商編號
        /// </summary>
        [LoadColumn(0)]
        public string? VendorId;

        /// <summary>
        /// 費率類型
        /// </summary>
        [LoadColumn(1)]
        public string? RateCode;

        /// <summary>
        /// 乘客數量
        /// </summary>
        [LoadColumn(2)]
        public float PassengerCount;

        /// <summary>
        /// 行程花費時間
        /// </summary>
        [LoadColumn(3)]
        public float TripTime;

        /// <summary>
        /// 行程距離
        /// </summary>
        [LoadColumn(4)]
        public float TripDistance;

        /// <summary>
        /// 付款方式
        /// </summary>
        [LoadColumn(5)]
        public string? PaymentType;

        /// <summary>
        /// 車資
        /// </summary>
        [LoadColumn(6)]
        public float FareAmount;
    }
}
