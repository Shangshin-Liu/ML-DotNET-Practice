using Microsoft.ML.Data;

namespace TaxiFarePrediction.Model
{
    /// <summary>
    /// 計程車車資預設結果模型
    /// </summary>
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
