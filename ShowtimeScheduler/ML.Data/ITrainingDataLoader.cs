using Microsoft.ML;

namespace ML.Data;

/// <summary>
/// Defines an interface for loading training data into an ML.NET IDataView.
/// Implementations can load data from various sources (file, files, databases etc.).
/// 
/// Src: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/load-data-ml-net
/// </summary>
public interface ITrainingDataLoader
{
    /// <summary>
    /// Loads the training data using the specified MLContext.
    /// </summary>
    /// <param name="mlContext">MLContext instance used for loading data.</param>
    /// <returns>IDataView containing the training data.</returns>
    IDataView LoadData(MLContext mlContext);
}