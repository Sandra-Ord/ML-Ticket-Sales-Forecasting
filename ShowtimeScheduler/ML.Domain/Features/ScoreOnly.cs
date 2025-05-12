using Microsoft.ML.Data;

namespace ML.Domain.Features;

public class ScoreOnly
{
    [ColumnName("Score")]
    public float Score { get; set; }
}