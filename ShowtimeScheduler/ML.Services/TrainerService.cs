using Microsoft.ML;
using ML.Data;
using ML.Model;
using ML.Domain.Mappings;

namespace ML.Services;

/// <summary>
/// Handles the training of ML.NET models and saving them to disk.
/// Relies on an <see cref="ITrainingDataLoader"/> to supply the training data
/// and uses a provided <see cref="MLContext"/> to build and save the model.
/// 
/// Train Src: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/train-machine-learning-model-ml-net
/// Save Src: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/save-load-machine-learning-models-ml-net
/// </summary>
public class TrainerService
{
    private readonly string _modelPath;
    private readonly MLContext _mlContext;
    private readonly ITrainingDataLoader _dataLoader;

    /// <summary>
    /// Initializes a new instance of the <see cref="TrainerService"/> class.
    /// </summary>
    /// <param name="dataLoader">Data loader responsible for providing training data.</param>
    /// <param name="modelPath">File path where the trained model will be saved.</param>
    public TrainerService(ITrainingDataLoader dataLoader, string modelPath)
    {
        _modelPath = modelPath;
        _mlContext = new MLContext();
        _dataLoader = dataLoader;
    }

    /// <summary>
    /// Trains a FastTreeTweedie regression model using the loaded training data and saves it to the configured file path.
    /// 
    /// Train Src: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/train-machine-learning-model-ml-net
    /// Save Src: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/save-load-machine-learning-models-ml-net
    /// </summary>
    public void TrainAndSaveModel()
    {
        Console.WriteLine("[TrainerService] Loading data for training...");
        var data = _dataLoader.LoadData(_mlContext);
        Console.WriteLine("[TrainerService] Finished data loading.");

        _mlContext.ComponentCatalog.RegisterAssembly(typeof(CyclicalTimeMapping).Assembly);
        _mlContext.ComponentCatalog.RegisterAssembly(typeof(ExistenceFlagMapper).Assembly);
        _mlContext.ComponentCatalog.RegisterAssembly(typeof(AverageAdmissionsMapper).Assembly);
        _mlContext.ComponentCatalog.RegisterAssembly(typeof(RatingMapper).Assembly);
        _mlContext.ComponentCatalog.RegisterAssembly(typeof(GenreMapper).Assembly);

        var pipeline = FastTreeDefinition.CreatePipeline(_mlContext);

        Console.WriteLine("[TrainerService] Starting model training...");
        var model = pipeline.Fit(data);
        Console.WriteLine("[TrainerService] Finished model training.");


        _mlContext.Model.Save(model, data.Schema, _modelPath);
    }

}