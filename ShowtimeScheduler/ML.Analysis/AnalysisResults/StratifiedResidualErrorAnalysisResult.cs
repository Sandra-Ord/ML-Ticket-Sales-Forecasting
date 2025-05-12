namespace ML.Analysis.AnalysisResults;

class StratifiedResidualErrorAnalysisResult
{
    public bool UseLog { get; set; }
    public double RSquared { get; set; }
    public double RMSE { get; set; }
    public double MAE { get; set; }

    public double MAPE { get; set; }

    // Only filled for log-transform case
    public double? R2_Original { get; set; }
    public double? RMSE_Original { get; set; }  
    public double? MAE_Original { get; set; }
}