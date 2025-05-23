using Microsoft.ML;
using Microsoft.ML.Data;
using ML.Domain;
using ML.Domain.Features;
using System.Data;
using System.Text;

namespace ML.Analysis;

/// <summary>
/// Provides functionality to train, evaluate, and export results from ML.NET regression models.
/// </summary>
public class ModelEvaluator
{
    private readonly MLContext _mlContext;
    private IDataView? _trainData;
    private IEstimator<ITransformer>? _modelPipeline;
    private ITransformer? _trainedModel;

    private string _labelColumnName = "Label";
    private string _scoreColumnName = "Score";

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelEvaluator"/> class with a default MLContext.
    /// </summary>
    public ModelEvaluator()
    {
        _mlContext = new MLContext();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelEvaluator"/> class with a custom MLContext.
    /// </summary>
    /// <param name="mlContext">Existing MLContext instance.<</param>
    public ModelEvaluator(MLContext mlContext)
    {
        _mlContext = mlContext;
    }

    /// <summary>
    /// Returns the trained model.
    /// </summary>
    /// <returns>Trained model.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the model is not trained.</exception>
    public ITransformer GetTrainedModel()
    {
        return _trainedModel ?? throw new InvalidOperationException("Model has not been trained yet.");
    }

    /// <summary>
    /// Sets the training data to be used for model training.
    /// </summary>
    /// <param name="trainData">IDataView training data.</param>
    /// <returns>Updated <see cref="ModelEvaluator"/> instance.</returns>
    public ModelEvaluator WithData(IDataView trainData)
    {
        _trainData = trainData;
        return this;
    }


    /// <summary>
    /// Assigns the ML.NET pipeline used to train the model.
    /// </summary>
    /// <param name="pipeline">Pipeline used to build the model.</param>
    /// <returns>Updated <see cref="ModelEvaluator"/> instance.</returns>
    public ModelEvaluator WithModel(IEstimator<ITransformer> pipeline)
    {
        _modelPipeline = pipeline;
        return this;
    }


    /// <summary>
    /// Trains the model using the provided data and pipeline.
    /// </summary>
    /// <param name="trainData">Optional training data to override the existing one.</param>
    /// <param name="pipeline">Optional pipeline to override the existing one.</param>
    /// <returns>Trained model as an <see cref="ITransformer"/>.</returns>
    public ITransformer BuildAndTrainModel(IDataView? trainData = null, IEstimator<ITransformer>? pipeline = null)
    {
        if (trainData != null) WithData(trainData);
        if (pipeline != null) WithModel(pipeline);

        if (_trainData == null)
            throw new InvalidOperationException("Training data must be provided.");
        if (_modelPipeline == null)
            throw new InvalidOperationException("Model pipeline must be provided.");

        _trainedModel = _modelPipeline.Fit(_trainData);
        return _trainedModel;
    }

    /// <summary>
    /// Applies the trained model to a new dataset.
    /// </summary>
    /// <param name="testData">IDataView test data to transform.</param>
    /// <returns>Transformed data.</returns>
    public IDataView TransformModel(IDataView testData)
    {
        EnsureModelIsTrained();
        return _trainedModel!.Transform(testData);
    }

    /// <summary>
    /// Evaluates the trained model using standard regression metrics and optionally writes results to file.
    /// </summary>
    /// <param name="testData">IDataView test dataset.</param>
    /// <param name="fileName">Optional filename to export predictions and metrics.</param>
    /// <returns>Regression metrics from ML.NET evaluation.</returns>
    public RegressionMetrics Evaluate(IDataView testData, string? fileName = null)
    {
        EnsureModelIsTrained();
        var predictions = _trainedModel!.Transform(testData);
        var metrics = _mlContext.Regression.Evaluate(predictions, _labelColumnName, _scoreColumnName);

        WriteMetricsToConsole(metrics.RSquared, metrics.RootMeanSquaredError, metrics.MeanAbsoluteError);

        if (!string.IsNullOrEmpty(fileName))
        {
            var truths = _mlContext.Data
                .CreateEnumerable<CinemaAdmissionData>(testData, false)
                .Select(r => r.Label)
                .ToArray();
            var preds = _mlContext.Data
                .CreateEnumerable<ScoreOnly>(predictions, false)
                .Select(r => r.Score)
                .ToArray();
            ResultsExporter.WriteTruthsAndPredictionsToFile(truths, preds.Select(Convert.ToDouble), fileName);
        }

        return metrics;
    }

    /// <summary>
    /// Evaluates the model with floored prediction values.
    /// </summary>
    /// <param name="testData">IDataView test dataset.</param>
    /// <returns>Tuple containing R², RMSE, and MAE metrics.</returns>
    public (double rSquared, double rmse, double mae) EvaluateFloored(IDataView testData)
    {
        return EvaluateWithCustomScoring(testData, scores => scores.Select(s => Math.Floor(s)).ToArray());
    }

    /// <summary>
    /// Evaluates the model and applies exponential normalization to predictions.
    /// Used when the target variable is log-transformed.
    /// Returns both original and normalized metric values.
    /// </summary>
    /// <param name="testData">IDataView test dataset.</param>
    /// <param name="fileName">Optional filename for exporting predictions.</param>
    /// <returns>A tuple with original and normalized R², RMSE, and MAE values.</returns>
    public (double rSquared, double rmse, double mae, double rSquaredOriginal, double rmseOriginal, double maeOriginal) EvaluateAndNormalize(IDataView testData, string? fileName = null)
    {
        EnsureModelIsTrained();

        var predictions = _trainedModel!.Transform(testData);
        var metrics = _mlContext.Regression.Evaluate(predictions, _labelColumnName, _scoreColumnName);

        var (r2Norm, rmseNorm, maeNorm) = EvaluateWithCustomScoring(
            testData,
            scores => scores.Select(s => Math.Exp(s) - 1).ToArray(),
            fileName,
            writeFile: !string.IsNullOrEmpty(fileName)
        );

        WriteNormalizedMetricsToConsole(
            metrics.RSquared, metrics.RootMeanSquaredError, metrics.MeanAbsoluteError,
            r2Norm, rmseNorm, maeNorm
        );

        return (
            metrics.RSquared, metrics.RootMeanSquaredError, metrics.MeanAbsoluteError,
            r2Norm, rmseNorm, maeNorm
        );
    }

    /// <summary>
    /// Evaluates the model and applies exponential normalization to predictions.
    /// Used when the target variable is log-transformed.
    /// Returns both original and normalized metric values.
    /// </summary>
    /// <param name="testData">IDataView test dataset.</param>
    /// <param name="fileName">Optional filename for exporting predictions.</param>
    /// <returns>A tuple with original and normalized R², RMSE, and MAE values.</returns>
    public (double rSquared, double rmse, double mae, double rSquaredOriginal, double rmseOriginal, double maeOriginal) EvaluateNormalizedFloored(IDataView testData, string? fileName = null)
    {
        EnsureModelIsTrained();

        var predictions = _trainedModel!.Transform(testData);
        var metrics = _mlContext.Regression.Evaluate(predictions, _labelColumnName, _scoreColumnName);

        var (r2Norm, rmseNorm, maeNorm) = EvaluateWithCustomScoring(
            testData,
            scores => scores.Select(s => Math.Exp(s) - 1).Select(s => Math.Floor(s)).ToArray(),
            fileName,
            writeFile: !string.IsNullOrEmpty(fileName)
        );

        WriteNormalizedMetricsToConsole(
            metrics.RSquared, metrics.RootMeanSquaredError, metrics.MeanAbsoluteError,
            r2Norm, rmseNorm, maeNorm
        );

        return (
            metrics.RSquared, metrics.RootMeanSquaredError, metrics.MeanAbsoluteError,
            r2Norm, rmseNorm, maeNorm
        );
    }

    /// <summary>
    /// Evaluates the model using a custom transformation on prediction scores.
    /// </summary>
    /// <param name="testData">IDataView test dataset.</param>
    /// <param name="scoreTransformer">Function to transform predicted scores before evaluation.</param>
    /// <param name="fileName">Optional file name for exporting data.</param>
    /// <param name="writeFile">Flag indicating whether to write the results to a file.</param>
    /// <returns>Tuple of R², RMSE, and MAE metrics.</returns>
    private (double r2, float rmse, float mae) EvaluateWithCustomScoring(
    IDataView testData,
    Func<float[], double[]> scoreTransformer,
    string? fileName = null,
    bool writeFile = false)
    {
        EnsureModelIsTrained();

        var predictions = _trainedModel!.Transform(testData);
        var truths = _mlContext.Data
            .CreateEnumerable<CinemaAdmissionData>(testData, false)
            .Select(r => r.Label)
            .ToArray();
        var scores = _mlContext.Data
            .CreateEnumerable<ScoreOnly>(predictions, false)
            .Select(r => r.Score)
            .ToArray();

        var preds = scoreTransformer(scores);
        var (r2, rmse, mae) = CalculateNormalScaleMetrics(truths, preds);

        if (writeFile && !string.IsNullOrEmpty(fileName))
        {
            ResultsExporter.WriteTruthsAndPredictionsToFile(truths, preds, fileName!);
        }

        return (r2, rmse, mae);
    }

    /// <summary>
    /// Calculates regression metrics based on predicted and true values.
    /// </summary>
    /// <param name="truths">Actual labels.</param>
    /// <param name="preds">Predicted values.</param>
    /// <returns>Tuple of R², RMSE, and MAE.</returns>
    private static (double r2, float rmse, float mae) CalculateNormalScaleMetrics(float[] truths, double[] preds)
    {
        float rmse = (float)Math.Sqrt(preds
            .Select((p, i) => Math.Pow(p - truths[i], 2))
            .Average());

        float mae = (float)preds
            .Select((p, i) => Math.Abs(p - truths[i]))
            .Average();

        var meanY = truths.Average();
        var sse = truths.Zip(preds, (y, p) => (y - p) * (y - p)).Sum();
        var sst = truths.Select(y => (y - meanY) * (y - meanY)).Sum();
        var r2 = 1 - sse / sst;

        return (r2, rmse, mae);
    }


    /// <summary>
    /// Writes both normalized and original metrics to the console.
    /// </summary>
    private static void WriteNormalizedMetricsToConsole(double r2, double rmse, double mae, double og_r2, double og_rmse, double og_mae)
    {
        Console.WriteLine("{0,-8:0.###}| {1,-10:0.###}| {2,-10:0.###}| {3,-8:0.###}| {4,-12:0.###}| {5,-10:0.###}",
            r2,
            rmse,
            mae,
            og_r2,
            og_rmse,
            og_mae);
    }

    /// <summary>
    /// Writes evaluation metrics to the console.
    /// </summary>
    private static void WriteMetricsToConsole(double r2, double rmse, double mae)
    {
        Console.WriteLine("{0,-8:0.##}| {1,-10:0.##}| {2,-10:0.##}",
            r2,
            rmse,
            mae);
    }

    /// <summary>
    /// Ensures the model has been trained before it is used.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the model is not trained.</exception>
    private void EnsureModelIsTrained()
    {
        if (_trainedModel == null)
            throw new InvalidOperationException("Model must be trained before evaluation.");
    }

}