using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;
using ML.Analysis;
using ML.Data;
using ML.Domain;
using ML.Model;
using System.Reflection;

string projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\..\.."));
string currentDirectory = Path.Combine(projectDirectory, "ML.Analysis");
string DataFolder = Path.Combine(currentDirectory, "Data");

string trainingData = Path.Combine(DataFolder, "train.csv");
string testingData = Path.Combine(DataFolder, "test.csv");
string allDataPath = Path.Combine(DataFolder, "ml_source_data.csv");


var theatreNames = new List<string>
{
    "Apollo Kino Astri",
    "Apollo Kino Coca-Cola Plaza",
    "Apollo Kino Eeden",
    "Apollo Kino Jõhvi",
    "Apollo Kino Kristiine",
    "Apollo Kino Lõunakeskus",
    "Apollo Kino Mustamäe",
    "Apollo Kino Pärnu",
    "Apollo Kino Saaremaa",
    "Apollo Kino Solaris",
    "Apollo Kino Tasku",
    "Apollo Kino Ülemiste"
};


MLContext mlContext = new MLContext(seed: 0);

//IDataView trainData = DataProvider.LoadFromFile(mlContext, trainingData);
//IDataView testData = DataProvider.LoadFromFile(mlContext, testingData);
IDataView allData = DataProvider.LoadFromFile(mlContext, allDataPath);


IEnumerable<CinemaAdmissionData> allDataEnumerable = mlContext.Data.CreateEnumerable<CinemaAdmissionData>(allData, reuseRowObject: false);

var trainTestSplitCutOffDate = new DateTime(2024, 2, 26);
IDataView trainData = mlContext.Data.LoadFromEnumerable(
    allDataEnumerable.Where(x => x.ShowDateTime.Date < trainTestSplitCutOffDate));
IDataView testData = mlContext.Data.LoadFromEnumerable(
    allDataEnumerable.Where(x => x.ShowDateTime.Date >= trainTestSplitCutOffDate));


//var trainer = new ModelEvaluator(mlContext);

// Fast Tree Tweedie Pipeline
//var pipeline = FastTreeTweedieDefinition.CreatePipeline(mlContext);

//mlContext.Data.CreateEnumerable<CinemaAdmissionData>(testData, reuseRowObject: false);





AnalysisHelper.BestFastTreeModel(mlContext, trainData, testData);

return;




// Thesis Chapter 5.1 - General Model Evaluation

AnalysisHelper.TrainAndEvaluateAllModels(mlContext, trainData, testData, true, 10);

AnalysisHelper.TrainAndEvaluateBestModels(mlContext, trainData, testData, true);

// Thesis Chapter 5.2 - Stratified Residual Error Analysis

//var ticketBasedBuckets = AnalysisHelper.GetColumnBasedBuckets(mlContext, testData, [0, 11, 26, 76, 151], "Label");

//Console.WriteLine("Fast Tree true");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTrainer(mlContext), trainData, ticketBasedBuckets, false, true, "Fast Tree");
//Console.WriteLine("Fast Tree false");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTrainer(mlContext), trainData, ticketBasedBuckets, false, false, "Fast Tree");
//Console.WriteLine("Fast Tree Tweedie true");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTweedieTrainer(mlContext), trainData, ticketBasedBuckets, false, true, "Fast Tree Tweedie");
//Console.WriteLine("Fast Tree Tweedie false");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTweedieTrainer(mlContext), trainData, ticketBasedBuckets, false, false, "Fast Tree Tweedie");
//Console.WriteLine("Light GBM true");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.LightGbmTrainer(mlContext), trainData, ticketBasedBuckets, false, true, "Light Gbm", 10);
//Console.WriteLine("Light GBM false");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.LightGbmTrainer(mlContext), trainData, ticketBasedBuckets, false, false, "Light Gbm", 10);


//var weekBasedBuckets = AnalysisHelper.GetColumnBasedBuckets(mlContext, testData, [double.MinValue, 1, 2, 3], "WeekNr");

//Console.WriteLine("Fast Tree true");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTrainer(mlContext), trainData, weekBasedBuckets, false, true, "Fast Tree");
//Console.WriteLine("Fast Tree false");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTrainer(mlContext), trainData, weekBasedBuckets, false, false, "Fast Tree");
//Console.WriteLine("Fast Tree Tweedie true");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTweedieTrainer(mlContext), trainData, weekBasedBuckets, false, true, "Fast Tree Tweedie");
//Console.WriteLine("Fast Tree Tweedie false");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTweedieTrainer(mlContext), trainData, weekBasedBuckets, false, false, "Fast Tree Tweedie");
//Console.WriteLine("Light GBM true");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.LightGbmTrainer(mlContext), trainData, weekBasedBuckets, false, true, "Light Gbm", 10);
//Console.WriteLine("Light GBM false");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.LightGbmTrainer(mlContext), trainData, weekBasedBuckets, false, false, "Light Gbm", 10);

// Thesis Chapter 5.3 - Hyperparameter Optimization

// Warning: May take a few hours to run.
AnalysisHelper.FastTreeHyperParameterGridSearch(mlContext, trainData, testData);

//var ticketBasedBuckets = AnalysisHelper.GetColumnBasedBuckets(mlContext, testData, [0, 11, 26, 76, 151], "Label");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, FastTreeDefinition.CreateTrainer(mlContext), trainData, ticketBasedBuckets, false, true, "Fast Tree");


//var weekBasedBuckets = AnalysisHelper.GetColumnBasedBuckets(mlContext, testData, [double.MinValue, 1, 2, 3], "WeekNr");
//AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, FastTreeDefinition.CreateTrainer(mlContext), trainData, weekBasedBuckets, false, true, "Fast Tree");


// Thesis Chapter 5.4 - Permutation Feature Importance Analysis

// Warning: May take a few hours to run
AnalysisHelper.PfiAnalysis(mlContext, trainData, testData, false, true, FastTreeDefinition.CreateTrainer(mlContext));



// Thesis Chapter 5.5 - Model Evaluation


// rolling origin
// measure before and after
// 
var dailyWindows = AnalysisHelper.GetRollingOriginData(mlContext, mlContext.Data.CreateEnumerable<CinemaAdmissionData>(allData, reuseRowObject: false), windowSizeDays: 1, numSplits: 10);
var biDailyWindows = AnalysisHelper.GetRollingOriginData(mlContext, mlContext.Data.CreateEnumerable<CinemaAdmissionData>(allData, reuseRowObject: false), windowSizeDays: 2, numSplits: 5);
var weeklyWindows = AnalysisHelper.GetRollingOriginData(mlContext, mlContext.Data.CreateEnumerable<CinemaAdmissionData>(allData, reuseRowObject: false), windowSizeDays: 7, numSplits: 5, lastTestStartDate: new DateTime(2024, 7, 8));
var biWeeklyWindows = AnalysisHelper.GetRollingOriginData(mlContext, mlContext.Data.CreateEnumerable<CinemaAdmissionData>(allData, reuseRowObject: false), windowSizeDays: 14, numSplits: 5, lastTestStartDate: new DateTime(2024, 7, 1));



AnalysisHelper.RollingOriginAnalysis(mlContext, ML.Model.FastTreeDefinition.CreateTrainer(mlContext), dailyWindows);

AnalysisHelper.RollingOriginAnalysis(mlContext, ML.Model.FastTreeDefinition.CreateTrainer(mlContext), biDailyWindows);

AnalysisHelper.RollingOriginAnalysis(mlContext, ML.Model.FastTreeDefinition.CreateTrainer(mlContext), weeklyWindows);

AnalysisHelper.RollingOriginAnalysis(mlContext, ML.Model.FastTreeDefinition.CreateTrainer(mlContext), biWeeklyWindows);





//AnalysisHelper.MeasureFastTreeTweedieModel(mlContext, trainData, testData);




return;