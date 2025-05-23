using Microsoft.ML;
using ML.Analysis;
using ML.Data;
using ML.Domain;
using ML.Model;

// ------------------------------------------- I M P O R T A N T -------------------------------------------

// This file is used for the analysis of the ML project.
// This file compiles all the different methods used in the analysis of the ML model.

// It is not meant to be run as a standalone application.
// Running this file in its entirety will take a long time and is not recommended,
// due to having to make changes in the PipelineBuilder.cs file (see comment before Thesis Chapter 5.5).

// The analysis is divided into different chapters, each chapter corresponds to a different section of the thesis.

// It is meant to be run in parts, depending on the analysis you want to perform,
// by manually setting the chapter in code or by using the console input.
// Additionally, you can further narrow down the analysis by commenting out the parts you don't want to run.

// ---------------------------------------------------------------------------------------------------------


// Set to false if you want to set the chapter manually from code
bool setChapterFromConsole = true; 

string? chapterToRun = null;
if (setChapterFromConsole)
{
    Console.WriteLine("Enter the chapter you want to run (e.g. 5.1, 5.2, 5.3, 5.4, 5.5):");
    chapterToRun = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(chapterToRun) || !"5.1,5.2,5.3,5.4,5.5".Contains(chapterToRun))
    {
        Console.WriteLine($"Invalid chapter '{chapterToRun}'. Exiting.");
        return;
    }
}
else
{
    // Set the chapter you want to run
    chapterToRun = "5.1"; 
}

#region Variable Declarations

string projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\..\.."));
string currentDirectory = Path.Combine(projectDirectory, "ML.Analysis");
string DataFolder = Path.Combine(currentDirectory, "Data");

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

// Seed fixed for reproducibility
MLContext mlContext = new MLContext(seed: 0);

IDataView allData = DataProvider.LoadFromFile(mlContext, allDataPath);
IEnumerable<CinemaAdmissionData> allDataEnumerable = mlContext.Data.CreateEnumerable<CinemaAdmissionData>(allData, reuseRowObject: false);

// 80/20 Train/Test Split
var trainTestSplitCutOffDate = new DateTime(2024, 2, 26);
IDataView trainData = mlContext.Data.LoadFromEnumerable(
    allDataEnumerable.Where(x => x.ShowDateTime.Date < trainTestSplitCutOffDate));
IDataView testData = mlContext.Data.LoadFromEnumerable(
    allDataEnumerable.Where(x => x.ShowDateTime.Date >= trainTestSplitCutOffDate));

// Stratified Data Buckets
var ticketBasedBuckets = AnalysisHelper.GetColumnBasedBuckets(mlContext, testData, [0, 11, 26, 76, 151], "Label");
var weekBasedBuckets = AnalysisHelper.GetColumnBasedBuckets(mlContext, testData, [double.MinValue, 1, 2, 3], "WeekNr");
var locationBasedBuckets = AnalysisHelper.GetLocationBasedBuckets(mlContext, testData, theatreNames);

// Rolling Origin Data Buckets
var weeklyWindows = AnalysisHelper.GetRollingOriginData(mlContext, allDataEnumerable, windowSizeDays: 7, numSplits: 30, lastTestStartDate: new DateTime(2024, 7, 8));

#endregion

#region Thesis Analysis

// ------------------------------------------------------------------------------- I M P O R T A N T -------------------------------------------------------------------------------
// Ensure the PipelineBuilder.cs file BuildPipelineNoNormalization method uses the CinemaAdmissionsFeatures.OriginalFeatureColumns() when calling the BuildBasePipeline() method.
// featureColumnNames: Features.OriginalFeatureColumns(),

switch (chapterToRun)
{
    #region Thesis Chapter 5.1 - Data Preparation
    case "5.1":

        AnalysisHelper.TrainAndEvaluateAllModels(mlContext, trainData, testData, true, 10);
        AnalysisHelper.TrainAndEvaluateBestModels(mlContext, trainData, testData, true);

        return;
    #endregion

    #region Thesis Chapter 5.2 - Stratified Residual Error Analysis
    case "5.2":

        Console.WriteLine("Fast Tree (log)");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTrainer(mlContext), trainData, ticketBasedBuckets, false, true, "Fast Tree");
        Console.WriteLine("Fast Tree");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTrainer(mlContext), trainData, ticketBasedBuckets, false, false, "Fast Tree");
        Console.WriteLine("Fast Tree Tweedie (log)");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTweedieTrainer(mlContext), trainData, ticketBasedBuckets, false, true, "Fast Tree Tweedie");
        Console.WriteLine("Fast Tree Tweedie");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTweedieTrainer(mlContext), trainData, ticketBasedBuckets, false, false, "Fast Tree Tweedie");
        Console.WriteLine("Light GBM (log)");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.LightGbmTrainer(mlContext), trainData, ticketBasedBuckets, false, true, "Light Gbm", 10);
        Console.WriteLine("Light GBM");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.LightGbmTrainer(mlContext), trainData, ticketBasedBuckets, false, false, "Light Gbm", 10);

        Console.WriteLine("Fast Tree (log)");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTrainer(mlContext), trainData, weekBasedBuckets, false, true, "Fast Tree");
        Console.WriteLine("Fast Tree");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTrainer(mlContext), trainData, weekBasedBuckets, false, false, "Fast Tree");
        Console.WriteLine("Fast Tree Tweedie (log)");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTweedieTrainer(mlContext), trainData, weekBasedBuckets, false, true, "Fast Tree Tweedie");
        Console.WriteLine("Fast Tree Tweedie");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.FastTreeTweedieTrainer(mlContext), trainData, weekBasedBuckets, false, false, "Fast Tree Tweedie");
        Console.WriteLine("Light GBM (log)");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.LightGbmTrainer(mlContext), trainData, weekBasedBuckets, false, true, "Light Gbm", 10);
        Console.WriteLine("Light GBM");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, AnalysisHelper.LightGbmTrainer(mlContext), trainData, weekBasedBuckets, false, false, "Light Gbm", 10);

        return;
    #endregion

    #region Thesis Chapter 5.3 - Hyperparameter Optimization
    case "5.3":

        // Warning: May take a long time to run.
        AnalysisHelper.FastTreeHyperParameterGridSearch(mlContext, trainData, testData);

        // Stratified Residual Error Analysis after parameter tuning
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, FastTreeDefinition.CreateTrainer(mlContext), trainData, ticketBasedBuckets, false, true, "Fast Tree");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, FastTreeDefinition.CreateTrainer(mlContext), trainData, weekBasedBuckets, false, true, "Fast Tree");

        return;
    #endregion

    #region Thesis Chapter 5.4 - (Permutation) Feature Importance Analysis
    case "5.4":
        
        // Warning: May take a long time to run
        AnalysisHelper.PfiAnalysis(mlContext, trainData, testData, false, true, FastTreeDefinition.CreateTrainer(mlContext));

        return;
    #endregion

    // ------------------------------------------------------------------------------- I M P O R T A N T -------------------------------------------------------------------------------
    // In PipelineBuilder.cs, change BuildPipelineNoNormalization method to use the CinemaAdmissionsFeatures.FeatureColumns() when calling the BuildBasePipeline() method.
    // featureColumnNames: Features.FeatureColumns(),
    // There is no equivalent change made in the BuildNormalizedPipeline method, as the normalization is not used in the further analysis.

    #region Thesis Chapter 5.5 - Evaluation
    case "5.5":

        // Stratified Residual Error Analysis
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, FastTreeDefinition.CreateTrainer(mlContext), trainData, ticketBasedBuckets, false, true, "Fast Tree");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, FastTreeDefinition.CreateTrainer(mlContext), trainData, weekBasedBuckets, false, true, "Fast Tree");
        AnalysisHelper.StratifiedResidualErrorAnalysis(mlContext, FastTreeDefinition.CreateTrainer(mlContext), trainData, locationBasedBuckets, false, true, "Fast Tree");

        // Rolling Origin Analysis
        AnalysisHelper.RollingOriginAnalysis(mlContext, FastTreeDefinition.CreateTrainer(mlContext), weeklyWindows);

        // Test for overfitting
        AnalysisHelper.MeasureModel(mlContext, trainData, testData, false, true, FastTreeDefinition.CreateTrainer(mlContext));
        AnalysisHelper.MeasureModel(mlContext, trainData, trainData, false, true, FastTreeDefinition.CreateTrainer(mlContext));

        return;
    #endregion

    default:
        return;
}
#endregion