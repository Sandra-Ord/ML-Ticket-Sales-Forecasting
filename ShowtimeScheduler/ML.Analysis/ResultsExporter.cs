using ML.Analysis.AnalysisResults;
using System.Text;

namespace ML.Analysis;

/// <summary>
/// Provides utility methods for exporting model analysis results and predictions to CSV files.
/// </summary>
public static class ResultsExporter
{
    /// <summary>
    /// Writes a CSV file with the specified filename (in the Results directory in the project root), header, and lines.
    /// </summary>
    /// <param name="fileName">Name of the file (with extension).</param>
    /// <param name="header">Header line of the CSV file.</param>
    /// <param name="lines">Data lines to include under the header.</param>
    public static void WriteCsv(string fileName, string header, IEnumerable<string> lines)
    {
        var resultsDir = Path.Combine(Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\..\..")), "Results");

        if (!Directory.Exists(resultsDir))
            Directory.CreateDirectory(resultsDir);

        var fullPath = Path.Combine(resultsDir, fileName);

        var csv = new StringBuilder();
        csv.AppendLine(header);
        foreach (var line in lines)
        {
            csv.AppendLine(line);
        }

        File.WriteAllText(fullPath, csv.ToString());
        Console.WriteLine($"Results written to {fullPath}");
    }

    /// <summary>
    /// Writes model evaluation results to a CSV file.
    /// </summary>
    /// <param name="fileName">Name of the file (with extension).</param>
    /// <param name="modelName">Display name of the evaluated model.</param>
    /// <param name="result">Evaluation results of the model.</param>
    /// <param name="includeOriginalScale">Whether to include metrics computed on the original scale of the data (in case the model used log-transformed labels).</param>
    public static void WriteModelEvaluation(string fileName, string modelName, ModelEvaluationResult result, bool includeOriginalScale = false)
    {
        var header = includeOriginalScale
            ? "ModelName;R2;RMSE;MAE;og_R2;og_RMSE;og_MAE;Time"
            : "ModelName;R2;RMSE;MAE;Time";

        var line = includeOriginalScale
            ? $"{modelName};{result.RSquared};{result.RMSE};{result.MAE};{result.RSquaredOriginalScale};{result.RMSEOriginalScale};{result.MAEOriginalScale};{result.TrainingTimeMilliseconds}"
            : $"{modelName};{result.RSquared};{result.RMSE};{result.MAE};{result.TrainingTimeMilliseconds}";

        WriteCsv(fileName, header, new[] { line });
    }

    /// <summary>
    /// Writes actual vs predicted values with residuals to a CSV file.
    /// </summary>
    /// <param name="truthsEnumerable">Enumerable of true values.</param>
    /// <param name="predsEnumerable">Enumerable of predicted values.</param>
    /// <param name="fileName">Filename to write to.</param>
    public static void WriteTruthsAndPredictionsToFile(IEnumerable<float> truthsEnumerable, IEnumerable<double> predsEnumerable, string fileName)
    {
        var truths = truthsEnumerable.ToArray();
        var preds = predsEnumerable.ToArray();

        var csv = new StringBuilder();
        csv.AppendLine("Actual;Predicted;Residual");
        for (int i = 0; i < truths.Length; i++)
        {
            csv.AppendLine($"{truths[i].ToString(System.Globalization.CultureInfo.InvariantCulture)};{preds[i].ToString(System.Globalization.CultureInfo.InvariantCulture)};{(truths[i] - preds[i]).ToString(System.Globalization.CultureInfo.InvariantCulture)}");
        }

        var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\..\.."));
        var resultsDirectory = Path.Combine(projectDirectory, "Results");

        if (!Directory.Exists(resultsDirectory))
        {
            Directory.CreateDirectory(resultsDirectory);
        }
        var filename = Path.Combine(resultsDirectory, $"{fileName.ToLower().Replace(" ", "_")}.csv");

        File.WriteAllText(filename, csv.ToString());
        Console.WriteLine($"Predictions saved to {filename}");
    }
}