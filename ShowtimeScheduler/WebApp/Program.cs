using Microsoft.ML;
using ML.Services;
using ML.Data;
using ML.Domain.Features;
using ML.Domain;
using Microsoft.Extensions.ML;

string projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\..\.."));

string modelPath = Path.Combine(projectDirectory, "model.zip");
string dataFolder = Path.Combine(projectDirectory, "Data");
string dataFilePath = Path.Combine(dataFolder, "TrainingData.csv");

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddHostedService<TrainingScheduler>();
//builder.Services.AddSingleton<ModelProvider>(new ModelProvider(modelPath)); 

builder.Services.AddSingleton<ITrainingDataLoader>(new FileTrainingDataLoader(dataFilePath));

builder.Services.AddSingleton<TrainerService>(serviceProvider =>
    new TrainerService(serviceProvider.GetRequiredService<ITrainingDataLoader>(), modelPath));

builder.Services.AddPredictionEnginePool<CinemaAdmissionData, ScoreOnly>()
       .FromFile(modelName: "AdmissionModel",
                 filePath: modelPath,
                 watchForChanges: true);

builder.Services.AddSingleton<InferenceEngine>();


builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

var trainer = app.Services.GetRequiredService<TrainerService>();
if (!File.Exists(modelPath))
{
    Console.WriteLine("[Startup] No model found. Training initial model...");
    trainer.TrainAndSaveModel();
}

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.MapControllers();
app.Run();