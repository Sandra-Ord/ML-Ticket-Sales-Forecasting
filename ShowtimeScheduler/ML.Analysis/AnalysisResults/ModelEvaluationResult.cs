namespace ML.Analysis.AnalysisResults;

public class ModelEvaluationResult
{
    public string ModelName { get; set; } = default!;
    public double RSquared { get; set; }
    public double RMSE { get; set; }
    public double MAE { get; set; }
    public double RSquaredOriginalScale { get; set; }
    public double RMSEOriginalScale { get; set; }
    public double MAEOriginalScale { get; set; }

    public double MAPE { get; set; }

    public long TrainingTimeMilliseconds { get; set; }

    public string? Slice { get; set; }
    public bool? Log { get; set; }

    public double? R2Mean { get; set; }
    public double? MAEMean { get; set; }
    public double? RMSEMean { get; set; }

    public double? R2StdDev { get; set; }
    public double? RMSEStdDev { get; set; }
    public double? MAEStdDev { get; set; }
}