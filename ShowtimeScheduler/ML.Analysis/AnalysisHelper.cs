using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using ML.Analysis.AnalysisResults;
using ML.Domain;
using ML.Model;
using System.Diagnostics;
using System.Globalization;

namespace ML.Analysis;

/// <summary>
/// Provides helper methods for training, evaluating, and analyzing ML.NET models,
/// including CSV export and analyses such as rolling origin cross validation, stratified error and PFI.
/// </summary>
class AnalysisHelper
{
    public static string ProjectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\..\.."));
    public static string ResultsDirectory = Path.Combine(ProjectDirectory, "Results");

    /// <summary>
    /// Trains a model with the specified trainer and pipeline, evaluates it on test data,
    /// and returns structured evaluation results. Optionally saves metrics to a CSV file.
    /// </summary>
    /// <param name="modelName">Display name of the model.</param>
    /// <param name="mlContext">MLContext instance to use.</param>
    /// <param name="trainData">IDataView training data.</param>
    /// <param name="testData">IDataView testing data.</param>
    /// <param name="trainer">ML.NET trainer algorithm.</param>
    /// <param name="normalized">Whether to normalize the features in the pipeline.</param>
    /// <param name="useLog">Whether to apply log-transform to the label.</param>
    /// <param name="saveToFile">Whether to save the evaluation metrics to a file.</param>
    /// <param name="fileName">Optional CSV filename (without extension) for saving metrics.</param>
    /// <returns><see cref="ModelEvaluationResult"/> containing evaluation metrics and timing.</returns>
    public static ModelEvaluationResult TrainAndEvaluate(
        string modelName,
        MLContext mlContext,
        IDataView trainData,
        IDataView testData,
        IEstimator<ITransformer> trainer,
        bool normalized,
        bool useLog,
        string[] featureColumns,
        bool saveToFile = false,
        string? fileName = null)
    {
        Console.WriteLine(modelName);

        var pipeline = PipelineBuilder.BuildPipeLine(mlContext, normalized, useLog, featureColumns)
            .Append(trainer);
        var modelTrainer = new ModelEvaluator(mlContext);

        var sw = Stopwatch.StartNew();
        modelTrainer.BuildAndTrainModel(trainData, pipeline);
        sw.Stop();

        Console.WriteLine("{0,-8}| {1,-10}| {2,-10}", "R^2", "RMSE", "MAE");

        fileName = saveToFile ? (String.IsNullOrEmpty(fileName) ? "modelName" + DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss") : fileName) : null;

        ModelEvaluationResult result;
        if (useLog)
        {
            var evals = saveToFile ? modelTrainer.EvaluateAndNormalize(testData, fileName) : modelTrainer.EvaluateAndNormalize(testData);

            result = new ModelEvaluationResult
                {
                    ModelName = modelName,
                    RSquared = evals.Item1,
                    RMSE = evals.Item2,
                    MAE = evals.Item3,
                    RSquaredOriginalScale = evals.Item4,
                    RMSEOriginalScale = evals.Item5,
                    MAEOriginalScale = evals.Item6,
                    TrainingTimeMilliseconds = sw.ElapsedMilliseconds,
                    Log = true,
                };
        } 
        else
        {
            var evals = saveToFile ? modelTrainer.Evaluate(testData, fileName) : modelTrainer.Evaluate(testData);

            result = new ModelEvaluationResult
            {
                ModelName = modelName,
                RSquaredOriginalScale = evals.RSquared,
                RMSEOriginalScale = evals.RootMeanSquaredError,
                MAEOriginalScale = evals.MeanAbsoluteError,
                TrainingTimeMilliseconds = sw.ElapsedMilliseconds,
                Log = false,
            };

        }

        Console.WriteLine();
        return result;
    }


    /// <summary>
    /// Train and Evaluate all models (with their default parameters) that ML.NET provides iterations times and find the average result.
    /// Used to gain a general understanding of how well the different models, generalize the data.
    /// It is important to use the same mlContext, training and testing data when establishing a baseline for model performances to even the playing ground.
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainData">IDataView training dataset.</param>
    /// <param name="testData">IDataView test dataset.</param>
    /// <param name="saveToFile">Whether to save the evaluation metrics to a file.</param>
    public static void TrainAndEvaluateAllModels(MLContext mlContext, IDataView trainData, IDataView testData, bool saveToFile = false, int iterations = 1)
    {
        var evaluationResults = new List<ModelEvaluationResult>();

        var aggregatedResults = new Dictionary<string, List<ModelEvaluationResult>>();

        void Accumulate(ModelEvaluationResult result)
        {
            var key = $"{result.ModelName} Log={result.Log}";

            if (!aggregatedResults.ContainsKey(key))
                aggregatedResults[key] = new List<ModelEvaluationResult>();
            aggregatedResults[key].Add(result);
        }

        var normalizedOutputColumns = CinemaAdmissionNormalizedFeatures.FeatureColumns();
        var outputColumns = CinemaAdmissionFeatures.OriginalFeatureColumns();

        // Deterministic when trained under these test conditions - to save time, only trained and evaluated once
        Accumulate(TrainAndEvaluateOgd(mlContext, trainData, testData, true, normalizedOutputColumns));
        Accumulate(TrainAndEvaluateOgd(mlContext, trainData, testData, false, normalizedOutputColumns));
        Accumulate(TrainAndEvaluateFastTree(mlContext, trainData, testData, true, outputColumns));
        Accumulate(TrainAndEvaluateFastTree(mlContext, trainData, testData, false, outputColumns));
        Accumulate(TrainAndEvaluateFastTreeTweedie(mlContext, trainData, testData, true, outputColumns));
        Accumulate(TrainAndEvaluateFastTreeTweedie(mlContext, trainData, testData, false, outputColumns));
        Accumulate(TrainAndEvaluateFastForest(mlContext, trainData, testData, true, outputColumns));
        Accumulate(TrainAndEvaluateFastForest(mlContext, trainData, testData, false, outputColumns));
        Accumulate(TrainAndEvaluateOls(mlContext, trainData, testData, true, normalizedOutputColumns));
        Accumulate(TrainAndEvaluateOls(mlContext, trainData, testData, false, normalizedOutputColumns));
        Accumulate(TrainAndEvaluateGam(mlContext, trainData, testData, true, normalizedOutputColumns));
        Accumulate(TrainAndEvaluateGam(mlContext, trainData, testData, false, normalizedOutputColumns));

        for (int i = 0; i < iterations; i++)
        {
            Accumulate(TrainAndEvaluateSdca(mlContext, trainData, testData, true, normalizedOutputColumns));
            Accumulate(TrainAndEvaluateSdca(mlContext, trainData, testData, false, normalizedOutputColumns));
            Accumulate(TrainAndEvaluateLbfgs(mlContext, trainData, testData, true, normalizedOutputColumns));
            Accumulate(TrainAndEvaluateLbfgs(mlContext, trainData, testData, false, normalizedOutputColumns));
            Accumulate(TrainAndEvaluateLightGbm(mlContext, trainData, testData, true, outputColumns));
            Accumulate(TrainAndEvaluateLightGbm(mlContext, trainData, testData, false, outputColumns));
        }

        // Averaging the results
        var finalResults = aggregatedResults.Select(kvp =>
        {
            var modelName = kvp.Key;
            var results = kvp.Value; // list of model evaluation results

            var meanMae = results.Average(r => r.MAEOriginalScale);
            var meanRmse = results.Average(r => r.RMSEOriginalScale);
            var meanr2 = results.Average(r => r.RSquared);
            var meanr2og = results.Average(r => r.RSquaredOriginalScale);
            return new ModelEvaluationResult
            {
                ModelName = modelName,

                RSquared = meanr2,
                RMSE = results.Average(r => r.RMSE),
                MAE = results.Average(r => r.MAE),
                RSquaredOriginalScale = meanr2og,
                RMSEOriginalScale = meanRmse,
                MAEOriginalScale = meanMae,
                TrainingTimeMilliseconds = (long)results.Average(r => r.TrainingTimeMilliseconds),

                MAEMean = meanMae,
                RMSEMean = meanRmse,

                RMSEStdDev = Math.Sqrt(results.Average(r => Math.Pow(r.RMSEOriginalScale - meanRmse, 2))),
                MAEStdDev = Math.Sqrt(results.Average(r => Math.Pow(r.MAEOriginalScale - meanMae, 2))),


            };
        }).ToList();

        if (!saveToFile) return;

        var lines = finalResults.Select(r =>
            $"{r.ModelName};{r.Log};{r.RSquared.ToString(CultureInfo.InvariantCulture)};{r.RMSE.ToString(CultureInfo.InvariantCulture)};{r.MAE.ToString(CultureInfo.InvariantCulture)};{r.RSquaredOriginalScale.ToString(CultureInfo.InvariantCulture)};{r.RMSEOriginalScale.ToString(CultureInfo.InvariantCulture)};{r.MAEOriginalScale.ToString(CultureInfo.InvariantCulture)};{r.TrainingTimeMilliseconds};{r.MAEMean};{r.RMSEMean};{r.MAEStdDev};{r.RMSEStdDev}");
        ResultsExporter.WriteCsv(
            $"all_model_evaluation_results_{DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss")}.csv",
            "ModelName;Log?;RSquared;RMSE;MAE;og_R2;og_RMSE;og_MAE;TrainingTimeMilliseconds;MaeMean;RmseMean;MaeStdDev;RmseStdDev",
            lines);
    }


    /// <summary>
    /// Trains and evaluates the best models (LightGBM, FastTree, and FastTreeTweedie) on the provided training and test datasets and optionally saves the results to a .csv file.
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainData">IDataView training dataset.</param>
    /// <param name="testData">IDataView test dataset.</param>
    /// <param name="saveToFile">Whether to save the evaluation metrics to a file.</param>
    public static void TrainAndEvaluateBestModels(MLContext mlContext, IDataView trainData, IDataView testData, bool saveToFile = false)
    {
        var evaluationResults = new List<ModelEvaluationResult>();
        var outputColumns = CinemaAdmissionFeatures.OriginalFeatureColumns();

        // ----------------------------------------------------------------------------------------------

        var eval = TrainAndEvaluateLightGbm(mlContext, trainData, testData, true, outputColumns);
        evaluationResults.Add(eval);
        eval = TrainAndEvaluateLightGbm(mlContext, trainData, testData, false, outputColumns);
        evaluationResults.Add(eval);

        // ----------------------------------------------------------------------------------------------

        eval = TrainAndEvaluateFastTree(mlContext, trainData, testData, true, outputColumns);
        evaluationResults.Add(eval);
        eval = TrainAndEvaluateFastTree(mlContext, trainData, testData, false, outputColumns);
        evaluationResults.Add(eval);

        // -------------------------------------------------------------------------------------------------------------------------

        eval = TrainAndEvaluateFastTreeTweedie(mlContext, trainData, testData, true, outputColumns);
        evaluationResults.Add(eval);
        eval = TrainAndEvaluateFastTreeTweedie(mlContext, trainData, testData, false, outputColumns);
        evaluationResults.Add(eval);

        //// -------------------------------------------------------------------------------------------------------------------------

        if (!saveToFile) return;

        var lines = evaluationResults.Select(r =>
            $"{r.ModelName};{r.RSquared.ToString(CultureInfo.InvariantCulture)};{r.RMSE.ToString(CultureInfo.InvariantCulture)};{r.MAE.ToString(CultureInfo.InvariantCulture)};{r.RSquaredOriginalScale.ToString(CultureInfo.InvariantCulture)};{r.RMSEOriginalScale.ToString(CultureInfo.InvariantCulture)};{r.MAEOriginalScale.ToString(CultureInfo.InvariantCulture)};{r.TrainingTimeMilliseconds}");
        ResultsExporter.WriteCsv(
            $"best_model_evaluation_results_{DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss")}.csv",
            "ModelName;RSquared;RMSE;MAE;og_R2;og_RMSE;og_MAE",
            lines);
    }


    /// <summary>
    /// Performs stratified residual error analysis on stratified subsets of the test data, using the model trained with the specified trainer and pipeline.
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainer">ML.NET trainer algorithm.</param>
    /// <param name="trainData">IDataView training data.</param>
    /// <param name="stratifiedTestData">Map of subgroup names and corresponding testing datasets.</param>
    /// <param name="normalized">Whether to normalize the features in the pipeline.</param>
    /// <param name="useLog">Whether to apply log-transform to the label.</param>
    /// <param name="modelName">Display name of the model.</param>
    /// <param name="saveToFile">Whether to save the evaluation metrics to a file.</param>
    /// <param name="fileName">Optional CSV filename (without extension) for saving metrics.</param>
    public static void StratifiedResidualErrorAnalysis(MLContext mlContext, IEstimator<ITransformer> trainer, IDataView trainData, Dictionary<string, IDataView> stratifiedTestData, bool normalized, bool useLog, string[] featureColumns, string modelName, int iterations = 1, bool saveToFile = false, string? fileName = null)
    {
        var aggregated = new Dictionary<string, List<StratifiedResidualErrorAnalysisResult>>();

        for (int i = 0; i < iterations; i++)
        {

            var pipeline = PipelineBuilder
            .BuildPipeLine(mlContext, normalized, useLog, featureColumns)
            .Append(trainer);

            var evaluator = new ModelEvaluator(mlContext);
            evaluator.BuildAndTrainModel(trainData, pipeline);
            foreach (var kvp in stratifiedTestData)
            {
                var sliceKey = kvp.Key;
                StratifiedResidualErrorAnalysisResult result;
                // NB! Ticket Based stratification - Do not consider the R2 values in the results, when creating buckets based on the label value, the mean value of the bucket is already a good indicator of the prediction

                if (useLog)
                {
                    var evals = saveToFile
                        ? evaluator.EvaluateAndNormalize(kvp.Value, fileName)
                        : evaluator.EvaluateAndNormalize(kvp.Value);

                    result = new StratifiedResidualErrorAnalysisResult
                    {
                        UseLog = useLog,
                        RSquared = evals.rSquared,
                        RMSE = evals.rmse,
                        MAE = evals.mae,
                        R2_Original = evals.rSquaredOriginal,
                        RMSE_Original = evals.rmseOriginal,
                        MAE_Original = evals.maeOriginal,
                    };
                }
                else
                {
                    var evals = saveToFile
                        ? evaluator.Evaluate(kvp.Value, fileName)
                        : evaluator.Evaluate(kvp.Value);
                    result = new StratifiedResidualErrorAnalysisResult
                    {
                        UseLog = useLog,
                        R2_Original = evals.RSquared,
                        RMSE_Original = evals.RootMeanSquaredError,
                        MAE_Original = evals.MeanAbsoluteError,
                    };
                }

                if (!aggregated.ContainsKey(sliceKey))
                    aggregated[sliceKey] = new List<StratifiedResidualErrorAnalysisResult>();

                aggregated[sliceKey].Add(result);
            }
        }

        var finalResults = aggregated.Select(kvp =>
        {
            var list = kvp.Value;

            var originalR2Mean = list.Average(r => r.R2_Original);
            var originalRmseMean = list.Average(r => r.RMSE_Original);
            var originalMaeMean = list.Average(r => r.MAE_Original);
            var r2mean = list.Average(r => r.RSquared);
            double stdR2 = Math.Sqrt(list.Average(r => Math.Pow(r.R2_Original!.Value - originalR2Mean!.Value, 2)));
            double stdRmse = Math.Sqrt(list.Average(r => Math.Pow(r.RMSE_Original!.Value - originalRmseMean!.Value, 2)));
            double stdMae = Math.Sqrt(list.Average(r => Math.Pow(r.MAE_Original!.Value - originalMaeMean!.Value, 2)));

            return new ModelEvaluationResult
            {
                Slice = kvp.Key,
                ModelName = modelName,
                Log = useLog,

                RSquared = r2mean,
                R2Mean = originalR2Mean!,
                RMSEMean = originalRmseMean!,
                MAEMean = originalMaeMean!,

                R2StdDev = stdR2,
                RMSEStdDev = stdRmse,
                MAEStdDev = stdMae
            };
        }).ToList();

        var outFile = fileName ??
            $"stratified_residual_error_analysis_{modelName.Replace(" ", "")}_{useLog}_{DateTime.Now:yyyy-MM-dd_HH-mm}.csv";

        var header = "Model;Slice;UseLog;LogR2;R2_Mean;RMSE_Mean;MAE_Mean;R2_StdDev;RMSE_StdDev;MAE_StdDev";
        var lines = finalResults.Select(r =>
            $"{r.ModelName};{r.Slice};{r.Log};{r.RSquared};{r.R2Mean};{r.RMSEMean};{r.MAEMean};{r.R2StdDev};{r.RMSEStdDev};{r.MAEStdDev}");

        ResultsExporter.WriteCsv(outFile, header, lines);
    }

    /// <summary>
    /// Performs a full Cartesian‐product hyperparameter grid search over FastTreeTweedie options,
    /// trains a model for each combination, evaluates (with and without log‐transform),
    /// and writes a single CSV summarizing leaves, trees, minExamples, learningRate, UseLog, R², RMSE, MAE, and original‐scale metrics where applicable.
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainData">IDataView training dataset.</param>
    /// <param name="testData">IDataView test dataset.</param>
    public static void FastTreeHyperParameterGridSearch(MLContext mlContext, IDataView trainData, IDataView testData, string[] featureColumns)
    {
        var numberOfLeavesOptions = new[] { 10, 20, 50, 100, 200 };
        var numberOfTreesOptions = new[] { 100, 200, 500, 1000 };
        var minExamplesPerLeafOptions = new[] { 1, 5, 10, 20, 50 };
        var learningRateOptions = new[] { 0.01f, 0.05f, 0.1f, 0.2f };
        var logOptions = new[] { true };

        var grid =
            from leaves in numberOfLeavesOptions
            from trees in numberOfTreesOptions
            from minPerLeaf in minExamplesPerLeafOptions
            from lr in learningRateOptions
            from useLog in logOptions
            select (leaves, trees, minPerLeaf, lr, useLog);

        var results = new List<GridSearchResult>();

        foreach (var (leaves, trees, minPerLeaf, lr, useLog) in grid)
        {
            var pipeline = PipelineBuilder.BuildPipeLine(mlContext, false, useLog, featureColumns)
                .Append(mlContext.Regression.Trainers.FastTree(numberOfLeaves: leaves, numberOfTrees: trees, minimumExampleCountPerLeaf: minPerLeaf, learningRate: lr));

            var modelTrainer = new ModelEvaluator(mlContext);

            modelTrainer.BuildAndTrainModel(trainData, pipeline);

            var result = new GridSearchResult
            {
                Leaves = leaves,
                Trees = trees,
                MinExamples = minPerLeaf,
                LearningRate = lr,
                UsedLogTransform = useLog,
            };

            if (useLog)
            {
                var evals = modelTrainer.EvaluateAndNormalize(testData);
                result.RSquared = evals.rSquared;
                result.RMSE = evals.rmse;
                result.MAE = evals.mae;
                result.R2_Original = evals.rSquaredOriginal;
                result.RMSE_Original = evals.rmseOriginal;
                result.MAE_Original = evals.maeOriginal;

            }
            else
            {
                var evals = modelTrainer.Evaluate(testData);
                result.R2_Original = evals.RSquared;
                result.RMSE_Original = evals.RootMeanSquaredError;
                result.MAE_Original = evals.MeanAbsoluteError;
            }

            results.Add(result);
        }

        var lines = results.Select(r =>
            $"FastTreeTweedie;{r.Leaves};{r.Trees};{r.MinExamples};{r.LearningRate};{r.UsedLogTransform};{r.RSquared};{r.RMSE};{r.MAE};{r.R2_Original};{r.RMSE_Original};{r.MAE_Original}");
        ResultsExporter.WriteCsv(
            "hyper_parameter_grid_search_fasttree.csv",
            "ModelName;Leaves;Trees;MinExamples;LearningRate;UseLog;RSquared;RMSE;MAE;og_R2;og_RMSE;og_MAE",
            lines);
    }

    /// <summary>
    /// Computes permutation feature importance (PFI) for a FastTreeTweedie model trained on <paramref name="trainData"/>.
    /// Extracts both signed and absolute ΔMAE and ΔR² per feature,
    /// prints them to the console, and writes two CSVs: one for absolute importances and one for signed deltas. 
    /// 
    /// Src: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/explain-machine-learning-model-permutation-feature-importance-ml-net
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainData">IDataView training dataset.</param>
    /// <param name="testData">IDataView test dataset.</param>
    public static void PfiAnalysis(MLContext mlContext, IDataView trainData, IDataView testData, bool normalized, bool logUsed, IEstimator<ITransformer> trainer, string[] featureColumns, int permutations = 5)
    {
        var pipeline = PipelineBuilder.BuildPipeLine(mlContext, normalized, logUsed, featureColumns).Append(trainer);

        var regressionTrainer = new ModelEvaluator(mlContext);

        regressionTrainer.BuildAndTrainModel(trainData, pipeline);

        var predictions = regressionTrainer.TransformModel(testData);

        string[] featureColumnNames = CinemaAdmissionFeatures.OriginalFeatureColumns();

        //// Get slot names from the model (Features that get encoded into slots (one-hot and multi-hot encoding))
        //// e.g. LanguageSpoken column gets encoded into slots LanguageSpoken.Estonian, LanguageSpoken.Russian, LanguageSpoken.English
        VBuffer<ReadOnlyMemory<char>> slotNames = default;
        predictions.Schema["Features"]
                        .GetSlotNames(ref slotNames);

        var slotNameArray = slotNames.DenseValues()
                                        .Select(x => x.ToString())
                                        .ToArray();

        // Perform the PFI analysis
        var pfiMetrics = mlContext.Regression.PermutationFeatureImportance(
            regressionTrainer.GetTrainedModel()!,
            predictions,
            labelColumnName: "Label",
            permutationCount: permutations
        );

        // Aggregate the PFI metrics to feature names
        var slotStats = pfiMetrics
            .Select(kvp => new
            {
                SlotName = kvp.Key,
                DeltaMAE = kvp.Value.MeanAbsoluteError.Mean,
                AbsDeltaMae = Math.Abs(kvp.Value.MeanAbsoluteError.Mean),
                DeltaR2 = kvp.Value.RSquared.Mean,
                AbsDelta = Math.Abs(kvp.Value.RSquared.Mean)
            }).ToArray();

        // Map to regroup cyclical time features
        var cyclicPairs = new Dictionary<string, string[]> {
            { "Month", new[] { "SinMonth", "CosMonth" } },
            { "DayOfMonth", new[] { "SinDayOfMonth", "CosDayOfMonth" } },
            { "DayOfYear", new[] { "SinDayOfYear", "CosDayOfYear" } },
            { "WeekDay", new[] { "SinWeekDay", "CosWeekDay" } },
            { "Hour", new[] { "SinHour", "CosHour" } },
            { "MinuteOfDay", new[] { "SinMinuteOfDay", "CosMinuteOfDay" } },
        };

        var signedFeatureDeltas = slotStats
            .GroupBy(s =>
            {
                // Detect if this is part of a cyclic feature
                foreach (var pair in cyclicPairs)
                {
                    if (pair.Value.Contains(s.SlotName.Split('.')[0]))
                        return pair.Key; // Group under the base name
                }
                // Otherwise, use the original feature
                return s.SlotName.Split('.')[0];
            })
            .Select(g => new
            {
                FeatureName = g.Key,
                SumDeltaMAE = g.Sum(x => x.DeltaMAE),
                MeanDeltaMae = g.Average(x => x.DeltaMAE),
                SumDeltaR2 = g.Sum(x => x.DeltaR2),
                MeanDeltaR2 = g.Average(x => x.DeltaR2)
            })
            .OrderByDescending(f => f.SumDeltaMAE)
            .ToList();


        var absFeatureImportances = slotStats
            .GroupBy(s =>
            {
                // Detect if this is part of a cyclic feature
                foreach (var pair in cyclicPairs)
                {
                    if (pair.Value.Contains(s.SlotName.Split('.')[0]))
                        return pair.Key; // Group under the base name
                }
                // Otherwise, use the original feature
                return s.SlotName.Split('.')[0];
            })
            .Select(g => new
            {
                FeatureName = g.Key,
                TotalImportanceMAE = g.Sum(x => x.AbsDeltaMae),
                MeanImportanceMAE = g.Average(x => x.AbsDeltaMae),
                TotalImportance = g.Sum(x => x.AbsDelta),
                MeanImportance = g.Average(x => x.AbsDelta)
            })
            .OrderByDescending(f => f.TotalImportanceMAE)
            .ToList();

        Console.WriteLine("=== Signed ΔMAE per Feature ===");
        foreach (var feature in signedFeatureDeltas)
            Console.WriteLine($"{feature.FeatureName};{feature.SumDeltaMAE};{feature.MeanDeltaMae};{feature.SumDeltaR2};{feature.MeanDeltaR2}");

        Console.WriteLine("\n=== Absolute Feature Importances ===");
        foreach (var feature in absFeatureImportances)
            Console.WriteLine($"{feature.FeatureName};{feature.TotalImportanceMAE};{feature.MeanImportanceMAE};{feature.TotalImportance};{feature.MeanImportance}");


        ResultsExporter.WriteCsv(
            "pfi_analysis_fasttree_absolute_pfi_mae.csv",
            "FeatureName;TotalImportanceMAE;MeanImportanceMAE;TotalImportance;MeanImportance",
            absFeatureImportances.Select(f =>
                $"{f.FeatureName};{f.TotalImportanceMAE};{f.MeanImportanceMAE};{f.TotalImportance};{f.MeanImportance}"));

        ResultsExporter.WriteCsv(
            "pfi_analysis_fasttree_signed_pfi_mae.csv",
            "FeatureName;SumDeltaMae;MeanDeltaMAE;SumDeltaR2;MeanDeltaR2",
            signedFeatureDeltas.Select(f =>
                $"{f.FeatureName};{f.SumDeltaMAE};{f.MeanDeltaMae};{f.SumDeltaR2};{f.MeanDeltaR2}"));
    }


    /// <summary>
    /// Performs rolling‐origin cross‐validation: for each pair of (train, test) IDataView in
    /// <paramref name="rollingOriginData"/>, trains the specified <paramref name="trainer"/> on the
    /// training split and evaluates on the test split. Writes out a CSV of R², RMSE, and MAE for each split.
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainer">ML.NET trainer algorithm.</param>
    /// <param name="rollingOriginData">Dictionary<IDataView, IDataView> with all the rolling origin training data (key) and testing data (value) pairs.</param>
    public static void RollingOriginAnalysis(MLContext mlContext, IEstimator<ITransformer> trainer, Dictionary<IDataView, IDataView> rollingOriginData, string[] featureColumns)
    {
        var normalized = false;
        var useLog = false;
        var pipeline = PipelineBuilder.BuildPipeLine(mlContext, normalized, useLog, featureColumns)
            .Append(trainer);

        var results = new List<StratifiedResidualErrorAnalysisResult>();
        foreach (var kvp in rollingOriginData)
        {
            var model = new ModelEvaluator(mlContext);
            model.BuildAndTrainModel(kvp.Key, pipeline);

            var evals = model.Evaluate(kvp.Value);
            results.Add(new StratifiedResidualErrorAnalysisResult
            {
                RSquared = evals.RSquared,
                RMSE = evals.RootMeanSquaredError,
                MAE = evals.MeanAbsoluteError,
            });
        }

        var lines = results.Select(r => $"{r.RSquared};{r.RMSE};{r.MAE}");
        ResultsExporter.WriteCsv(
            $"rolling_origin_analysis_{DateTime.Now.ToString("yyyy-MM-dd-HH-m-s")}.csv",
            "R2;RMSE;MAE",
            lines);
    }


    /// <summary>
    /// Trains a FastTree model using the specified MLContext and data, evaluates it on test data,
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainData">IDataView training dataset.</param>
    /// <param name="testData">IDataView test dataset.</param>
    public static void BestFastTreeModel(MLContext mlContext, IDataView trainData, IDataView testData, string[] featureColumns)
    {
        //var pipeline = FastTreeDefinition.CreatePipeline(mlContext);
        var pipeline = PipelineBuilder.BuildPipeLine(mlContext, false, true, featureColumns)
            .Append(FastTreeDefinition.CreateTrainer(mlContext));

        //var pipeline = PipelineBuilder.BuildPipeLine(mlContext, false, true).Append(FastTreeTrainer(mlContext));
        var modelTrainer = new ModelEvaluator(mlContext);

        modelTrainer.BuildAndTrainModel(trainData, pipeline);

        modelTrainer.EvaluateNormalizedFloored(testData);
    }

    /// <summary>
    /// Trains the FastTreeTweedie model and saves it to the specified path.
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainData">IDataView training dataset.</param>
    /// <param name="testData">IDataView test dataset.</param>
    /// <param name="modelPath">File path to save the trained model.</param>
    public static void SaveFastTreeTweedieModel(MLContext mlContext, IDataView trainData, IDataView testData, string[] featureColumns, string modelPath)
    {
        var pipeline = PipelineBuilder.BuildPipeLine(mlContext, false, true, featureColumns).Append(FastTreeDefinition.CreateTrainer(mlContext));
        var modelTrainer = new ModelEvaluator(mlContext);
        modelTrainer.BuildAndTrainModel(trainData, pipeline);

        var evals = modelTrainer.Evaluate(testData);

        mlContext.Model.Save(modelTrainer.GetTrainedModel(), trainData.Schema, modelPath);
    }

    /// <summary>
    /// Measures training time and evaluation metrics for the FastTreeTweedie model that is the most suitable for the data set,
    /// optionally writing results to CSV.
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainData">IDataView training dataset.</param>
    /// <param name="testData">IDataView test dataset.</param>
    /// <param name="fileName">Optional CSV filename (without extension) for output.</param>
    public static void MeasureModel(MLContext mlContext, IDataView trainData, IDataView testData, bool normalized, bool logUsed, IEstimator<ITransformer> trainer, string[] featureColumns, string? fileName = null)
    {
        //var pipeline = FastTreeDefinition.CreatePipeline(mlContext);
        var pipeline = PipelineBuilder.BuildPipeLine(mlContext, normalized, logUsed, featureColumns).Append(trainer);
        var modelTrainer = new ModelEvaluator(mlContext);

        var sw = Stopwatch.StartNew();
        modelTrainer.BuildAndTrainModel(trainData, pipeline);
        sw.Stop();

        var evals = modelTrainer.EvaluateAndNormalize(testData);

        ResultsExporter.WriteModelEvaluation(
            string.IsNullOrEmpty(fileName) ? $"time_data_best_{DateTime.Now:yyyy-MM-dd_HH-mm}.csv" : $"{fileName}.csv",
            "FastTree",
            new ModelEvaluationResult
            {
                RSquared = evals.rSquared,
                RMSE = evals.rmse,
                MAE = evals.mae,
                RSquaredOriginalScale = evals.rSquaredOriginal,
                RMSEOriginalScale = evals.rmseOriginal,
                MAEOriginalScale = evals.maeOriginal,
                TrainingTimeMilliseconds = sw.ElapsedMilliseconds
            }, true);
    }

    /// <summary>
    /// Generates CSV data for residual‐vs‐actual plots for each of several predefined models.
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="trainData">IDataView training dataset.</param>
    /// <param name="testData">IDataView test dataset.</param>
    public static void GenerateResidualPlotData(MLContext mlContext, IDataView trainData, IDataView testData)
    {

        TrainAndEvaluateLightGbm(mlContext, trainData, testData, true, CinemaAdmissionFeatures.OriginalFeatureColumns(), true, "Residual_Plot_Light_GBM_log");
        TrainAndEvaluateLightGbm(mlContext, trainData, testData, false, CinemaAdmissionFeatures.OriginalFeatureColumns(), true, "Residual_Plot_Light_GBM");
        TrainAndEvaluateFastTree(mlContext, trainData, testData, true, CinemaAdmissionFeatures.OriginalFeatureColumns(), true, "Residual_Plot_Fast_Tree_Log");
        TrainAndEvaluateFastTree(mlContext, trainData, testData, false, CinemaAdmissionFeatures.OriginalFeatureColumns(), true, "Residual_Plot_Fast_Tree");
        TrainAndEvaluateFastTreeTweedie(mlContext, trainData, testData, true, CinemaAdmissionFeatures.OriginalFeatureColumns(), true, "Residual_Plot_Fast_Tree_Tweedie_Log");
        TrainAndEvaluateFastTreeTweedie(mlContext, trainData, testData, false, CinemaAdmissionFeatures.OriginalFeatureColumns(), true, "Residual_Plot_Fast_Tree_Tweedie");
    }

    public static ModelEvaluationResult TrainAndEvaluateSdca(MLContext mlContext, IDataView trainData, IDataView testData, bool useLog, string[] featureColumns, bool saveToFile = false, string? fileName = null) =>
        TrainAndEvaluate("SDCA_Regression", mlContext, trainData, testData, SdcaTrainer(mlContext), true, useLog, featureColumns, saveToFile, fileName);

    public static ModelEvaluationResult TrainAndEvaluateLbfgs(MLContext mlContext, IDataView trainData, IDataView testData, bool useLog, string[] featureColumns, bool saveToFile = false, string? fileName = null) =>
        TrainAndEvaluate("Lbfgs_Poisson_Regression", mlContext, trainData, testData, LbfgsPoissonTrainer(mlContext), true, useLog, featureColumns, saveToFile, fileName);

    public static ModelEvaluationResult TrainAndEvaluateOgd(MLContext mlContext, IDataView trainData, IDataView testData, bool useLog, string[] featureColumns, bool saveToFile = false, string? fileName = null) =>
        TrainAndEvaluate("OGD Regression", mlContext, trainData, testData, OgdTrainer(mlContext), true, useLog, featureColumns, saveToFile, fileName);

    public static ModelEvaluationResult TrainAndEvaluateLightGbm(MLContext mlContext, IDataView trainData, IDataView testData, bool useLog, string[] featureColumns, bool saveToFile = false, string? fileName = null) =>
        TrainAndEvaluate("Light_GBM", mlContext, trainData, testData, LightGbmTrainer(mlContext), false, useLog, featureColumns, saveToFile, fileName);

    public static ModelEvaluationResult TrainAndEvaluateFastTree(MLContext mlContext, IDataView trainData, IDataView testData, bool useLog, string[] featureColumns, bool saveToFile = false, string? fileName = null) =>
        TrainAndEvaluate("Fast_Tree_Regression", mlContext, trainData, testData, FastTreeTrainer(mlContext), false, useLog, featureColumns, saveToFile, fileName);

    public static ModelEvaluationResult TrainAndEvaluateFastTreeTweedie(MLContext mlContext, IDataView trainData, IDataView testData, bool useLog, string[] featureColumns, bool saveToFile = false, string? fileName = null) =>
        TrainAndEvaluate("Fast_Tree_Tweedie_Regression", mlContext, trainData, testData, FastTreeTweedieTrainer(mlContext), false, useLog, featureColumns, saveToFile, fileName);

    public static ModelEvaluationResult TrainAndEvaluateFastForest(MLContext mlContext, IDataView trainData, IDataView testData, bool useLog, string[] featureColumns, bool saveToFile = false, string? fileName = null) =>
        TrainAndEvaluate("Fast_Forest_Regression", mlContext, trainData, testData, FastForestTrainer(mlContext), false, useLog, featureColumns, saveToFile, fileName);

    public static ModelEvaluationResult TrainAndEvaluateOls(MLContext mlContext, IDataView trainData, IDataView testData, bool useLog, string[] featureColumns, bool saveToFile = false, string? fileName = null) =>
        TrainAndEvaluate("OLS_Regression", mlContext, trainData, testData, OlsTrainer(mlContext), true, useLog, featureColumns, saveToFile, fileName);

    public static ModelEvaluationResult TrainAndEvaluateGam(MLContext mlContext, IDataView trainData, IDataView testData, bool useLog, string[] featureColumns, bool saveToFile = false, string? fileName = null) =>
        TrainAndEvaluate("Gam_Regression", mlContext, trainData, testData, GamTrainer(mlContext), true, useLog, featureColumns, saveToFile, fileName);

    public static IEstimator<ITransformer> SdcaTrainer(MLContext mlContext) => mlContext.Regression.Trainers.Sdca();
    public static IEstimator<ITransformer> LbfgsPoissonTrainer(MLContext mlContext) => mlContext.Regression.Trainers.Sdca();
    public static IEstimator<ITransformer> OgdTrainer(MLContext mlContext) => mlContext.Regression.Trainers.OnlineGradientDescent();
    public static IEstimator<ITransformer> LightGbmTrainer(MLContext mlContext) => mlContext.Regression.Trainers.LightGbm();
    public static IEstimator<ITransformer> FastTreeTrainer(MLContext mlContext) => mlContext.Regression.Trainers.FastTree();
    public static IEstimator<ITransformer> FastTreeTweedieTrainer(MLContext mlContext) => mlContext.Regression.Trainers.FastTreeTweedie();
    public static IEstimator<ITransformer> FastForestTrainer(MLContext mlContext) => mlContext.Regression.Trainers.FastForest();
    public static IEstimator<ITransformer> OlsTrainer(MLContext mlContext) => mlContext.Regression.Trainers.Ols();
    public static IEstimator<ITransformer> GamTrainer(MLContext mlContext) => mlContext.Regression.Trainers.Gam();


    public static Dictionary<string, IDataView> GetColumnBasedBuckets(MLContext mlContext, IDataView testData, List<double> splitPoints, string columnName)
    {
        splitPoints = splitPoints.OrderBy(s => s).ToList();
        var testBuckets = new Dictionary<string, IDataView>
        {
            { "All", testData }
        };

        for (int i = 0; i < splitPoints.Count - 1; i++)
        {
            double lower = splitPoints[i];
            double upper = splitPoints[i + 1];
            string label = $"{(int) lower}–{(int) (upper - 1)}";

            var filtered = mlContext.Data.FilterRowsByColumn(testData, columnName, lower, upper);
            testBuckets[label] = filtered;
        }

        double lastLower = splitPoints[^1];
        string lastLabel = $"{(int)lastLower}+";
        var finalFiltered = mlContext.Data.FilterRowsByColumn(testData, columnName, lowerBound: lastLower);
        testBuckets[lastLabel] = finalFiltered;

        return testBuckets;
    }

    public static Dictionary<string, IDataView> GetLocationBasedBuckets(MLContext mlContext, IDataView testData, List<string> theatreNames)
    {
        var testDataEnumerable = mlContext.Data.CreateEnumerable<CinemaAdmissionData>(testData, reuseRowObject: false);
        var testBuckets = new Dictionary<string, IDataView>
        {
            { "All", mlContext.Data.LoadFromEnumerable(testDataEnumerable) }
        };
        foreach (var theatreName in theatreNames)
        {
            var filteredData = testDataEnumerable.Where(s => s.TheatreName == theatreName);
            testBuckets[theatreName] = mlContext.Data.LoadFromEnumerable(filteredData);
        }

        return testBuckets;
    }

    public static Dictionary<IDataView, IDataView> GetRollingOriginData(MLContext mlContext, IEnumerable<CinemaAdmissionData> allData, int windowSizeDays, int numSplits, DateTime? lastTestStartDate = null)
    {
        var sorted = allData
            .OrderBy(x => x.ShowDateTime)
            .ToList();
        if (sorted.Count == 0)
            return new Dictionary<IDataView, IDataView>();

        DateTime maxDate = allData.Last().ShowDateTime.Date;
        DateTime finalTestStart = lastTestStartDate?.Date
                                  ?? maxDate.AddDays(-windowSizeDays + 1);

        if (finalTestStart > maxDate)
            throw new ArgumentException(
               $"finalTestStart ({finalTestStart:d}) is after last data date ({maxDate:d}).");

        DateTime firstTestStart = finalTestStart
            .AddDays(-windowSizeDays * (numSplits - 1));

        //var dict = new Dictionary<string, (IDataView Train, IDataView Test)>();

        var testBuckets = new Dictionary<IDataView, IDataView>();
        for (int i = 0; i < numSplits; i++)
        {
            // Test window
            var testStart = firstTestStart.AddDays(windowSizeDays * i);
            var testEnd = testStart.AddDays(windowSizeDays - 1);

            // TestEnd is no bigger than maxDate
            if (testEnd > maxDate)
                testEnd = maxDate;

            // c) Training goes up to the day before testStart
            var trainEnd = testStart.AddDays(-1);

            var trainEnumerable = allData
                .Where(x => x.ShowDateTime.Date <= trainEnd);
            var testEnumerable = allData
                .Where(x => x.ShowDateTime.Date >= testStart
                         && x.ShowDateTime.Date <= testEnd);

            string label = $"{testStart:yyyy-MM-dd}→{testEnd:yyyy-MM-dd}";
            var trainDv = mlContext.Data.LoadFromEnumerable(trainEnumerable);
            var testDv = mlContext.Data.LoadFromEnumerable(testEnumerable);

            testBuckets[trainDv] = testDv;
            //dict[label] = (trainDv, testDv);
        }
        return testBuckets;
    }
    
}