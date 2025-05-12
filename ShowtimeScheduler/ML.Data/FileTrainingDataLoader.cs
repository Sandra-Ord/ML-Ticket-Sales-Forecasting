using Microsoft.ML;
using ML.Domain;

namespace ML.Data;

/// <summary>
/// Loads training data from a delimited text file (.csv).
/// 
/// Src: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/load-data-ml-net
/// </summary>
public class FileTrainingDataLoader : ITrainingDataLoader
{
    private readonly string _filePath;

    /// <summary>
    /// Initializes a new instance of the <see cref="FileTrainingDataLoader"/> class.
    /// </summary>
    /// <param name="filePath">Path to the training data file.</param>
    public FileTrainingDataLoader(string filePath)
    {
        _filePath = filePath;
    }

    /// <summary>
    /// Loads data from the specified file path into an IDataView.
    /// </summary>
    /// <param name="mlContext">MLContext used to create the IDataView.</param>
    /// <returns>IDataView containing the training data from the file.</returns>
    /// <exception cref="FileNotFoundException">Thrown if the file does not exist.</exception>
    public IDataView LoadData(MLContext mlContext)
    {
        if (!File.Exists(_filePath))
            throw new FileNotFoundException($"Training data file not found: {_filePath}");

        Console.WriteLine($"[FileLoader] Loading data from file: {_filePath}");

        return mlContext.Data.LoadFromTextFile<CinemaAdmissionData>(
            path: _filePath,
            hasHeader: true,
            separatorChar: ';'
        );
    }
}
