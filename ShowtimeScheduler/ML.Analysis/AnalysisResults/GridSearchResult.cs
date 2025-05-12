namespace ML.Analysis.AnalysisResults;

class GridSearchResult
{
    public int Leaves { get; set; }
    public int Trees { get; set; }
    public int MinExamples { get; set; }
    public float LearningRate { get; set; }
    public bool UsedLogTransform { get; set; }

    public double RSquared { get; set; }
    public double RMSE { get; set; }
    public double MAE { get; set; }

    // Only filled for log-transform case
    public double? R2_Original { get; set; }
    public double? RMSE_Original { get; set; }  
    public double? MAE_Original { get; set; }
}