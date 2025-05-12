using Microsoft.ML;
using Microsoft.ML.Transforms;
using ML.Domain.Features;
using ML.Domain.Mappings;
using CData = ML.Domain.CinemaAdmissionData;
using Features = ML.Domain.CinemaAdmissionFeatures;
using NormalizedFeatures = ML.Domain.CinemaAdmissionNormalizedFeatures;
using Averages = ML.Domain.Features.AverageTimePeriodAdmissions;
using System.Data;

namespace ML.Model;

/// <summary>
/// PipelineBuilder is responsible for building the ML.NET data transformation pipeline for the model.
/// Used for training and inference of cinema admission predictions.
/// </summary>
public static class PipelineBuilder
{

    private static readonly (string input, string output)[] _NormalizeMap = new (string input, string output)[]
    {
        (nameof(CData.LengthInMinutes),                     nameof(NormalizedFeatures.LengthInMinutesNorm)),
        (nameof(EventOrdinalRating.EventRatingEncoded),     nameof(NormalizedFeatures.EventRatingEncodedNorm)),
        (nameof(CData.WeekNr),                              nameof(NormalizedFeatures.WeekNrNorm)),
        (nameof(CyclicalTime.Year),                         nameof(NormalizedFeatures.YearNorm)),

        (nameof(CData.AverageAdmissions),               nameof(NormalizedFeatures.AverageAdmissions)),
        (nameof(CData.TotalAverageAdmissions),          nameof(NormalizedFeatures.TotalAverageAdmissions)),

        (nameof(CData.PreViewAdmissions),               nameof(NormalizedFeatures.PreViewAdmissionsNorm)),
        (nameof(CData.PreViewShows),                    nameof(NormalizedFeatures.PreViewShowsNorm)),
        (nameof(Averages.PreViewAverageAdmissions),     nameof(NormalizedFeatures.PreViewAverageAdmissionsNorm)),

        (nameof(CData.TotalPreViewAdmissions),              nameof(NormalizedFeatures.TotalPreViewAdmissionsNorm)),
        (nameof(CData.TotalPreViewShows),                   nameof(NormalizedFeatures.TotalPreViewShowsNorm)),
        (nameof(Averages.TotalPreViewAverageAdmissions),    nameof(NormalizedFeatures.TotalPreViewAverageAdmissionsNorm)),

        (nameof(CData.FirstWeekAdmissions),             nameof(NormalizedFeatures.FirstWeekAdmissionsNorm)),
        (nameof(CData.FirstWeekShows),                  nameof(NormalizedFeatures.FirstWeekShowsNorm)),
        (nameof(Averages.FirstWeekAverageAdmissions),   nameof(NormalizedFeatures.FirstWeekAverageAdmissionsNorm)),

        (nameof(CData.TotalFirstWeekAdmissions),            nameof(NormalizedFeatures.TotalFirstWeekAdmissionsNorm)),
        (nameof(CData.TotalFirstWeekShows),                 nameof(NormalizedFeatures.TotalFirstWeekShowsNorm)),
        (nameof(Averages.TotalFirstWeekAverageAdmissions),  nameof(NormalizedFeatures.TotalFirstWeekAverageAdmissionsNorm)),

        (nameof(CData.LastWeekEndAdmissions),           nameof(NormalizedFeatures.LastWeekEndAdmissionsNorm)),
        (nameof(CData.LastWeekEndShows),                nameof(NormalizedFeatures.LastWeekEndShowsNorm)),
        (nameof(Averages.LastWeekEndAverageAdmissions), nameof(NormalizedFeatures.LastWeekEndAverageAdmissionsNorm)),

        (nameof(CData.LastWeekAdmissions),              nameof(NormalizedFeatures.LastWeekAdmissionsNorm)),
        (nameof(CData.LastWeekShows),                   nameof(NormalizedFeatures.LastWeekShowsNorm)),
        (nameof(Averages.LastWeekAverageAdmissions),    nameof(NormalizedFeatures.LastWeekAverageAdmissionsNorm)),

        (nameof(CData.TotalLastWeekAdmissions),             nameof(NormalizedFeatures.TotalLastWeekAdmissionsNorm)),
        (nameof(CData.TotalLastWeekShows),                  nameof(NormalizedFeatures.TotalLastWeekShowsNorm)),
        (nameof(Averages.TotalLastWeekAverageAdmissions),   nameof(NormalizedFeatures.TotalLastWeekAverageAdmissionsNorm)),

        (nameof(CData.PreLastWeekAdmissions),           nameof(NormalizedFeatures.PreLastWeekAdmissionsNorm)),
        (nameof(CData.PreLastWeekShows),                nameof(NormalizedFeatures.PreLastWeekShowsNorm)),
        (nameof(Averages.PreLastWeekAverageAdmissions), nameof(NormalizedFeatures.PreLastWeekAverageAdmissionsNorm)),

        (nameof(CData.TotalPreLastWeekAdmissions),              nameof(NormalizedFeatures.TotalPreLastWeekAdmissionsNorm)),
        (nameof(CData.TotalPreLastWeekShows),                   nameof(NormalizedFeatures.TotalPreLastWeekShowsNorm)),
        (nameof(Averages.TotalPreLastWeekAverageAdmissions),    nameof(NormalizedFeatures.TotalPreLastWeekAverageAdmissionsNorm)),
    };

    /// <summary>
    /// Builds an ML.NET pipeline, optionally including normalization of numerical features and logarithmic transformation of the label.
    /// </summary>
    /// <param name="mlContext">MLContext to use</param>
    /// <param name="normalized">If true, applies normalization to numeric input features.</param>
    /// <param name="logUsed">If true, applies logarithmic transformation to labels.</param>
    /// <returns>ML.NET data transformation pipeline.</returns>
    public static IEstimator<ITransformer> BuildPipeLine(MLContext mlContext, bool normalized, bool logUsed)
    {
        if (normalized)
        {
            return BuildNormalizedPipeline(mlContext, logUsed);
        }
        else
        {
            return BuildPipelineNoNormalization(mlContext, logUsed);
        }
    }

    /// <summary>
    /// Builds an ML.NET pipeline without normalization of numerical features.
    /// </summary>
    /// <param name="mlContext">MLContext to use.</param>
    /// <param name="logUsed">If true, applies logarithmic transformation to labels.</param>
    /// <returns>Unnormalized data pipeline.</returns>
    public static IEstimator<ITransformer> BuildPipelineNoNormalization(MLContext mlContext, bool logUsed = true)
    {
        return BuildBasePipeline(
            mlContext,
            //eventNameFeaturizedOutputColumnName: nameof(Features.EventNameFeaturized),
            oneHotEncodedColumnNames: new (string output, string input)[]
            {
                (nameof(Features.TheatreNameEncoded), nameof(CData.TheatreName)),
                (nameof(Features.CityEncoded), nameof(CData.City)),
                (nameof(Features.CountryEncoded), nameof(CData.Country)),
                (nameof(Features.EventTypeEncoded), nameof(CData.EventType)),
                (nameof(Features.SpokenLanguageEncoded), nameof(CData.SpokenLanguage)),
                (nameof(Features.PresentationMethodEncoded), nameof(CData.PresentationMethod)),
            },
            eventGenresEncodedOutputColumnName: nameof(Features.EventGenresEncoded),
            featureColumnNames: Features.FeatureColumns(),
            logUsed: logUsed,
            normalizeColumnNames: null
        );
    }

    /// <summary>
    /// Builds a pipeline with MinMax normalization applied to specified numeric features.
    /// </summary>
    /// <param name="mlContext">MLContext to use.</param>
    /// <param name="logUsed">If true, applies logarithmic transformation to labels.</param>
    /// <returns>Normalized data pipeline.</returns>
    public static IEstimator<ITransformer> BuildNormalizedPipeline(MLContext mlContext, bool logUsed = true)
    {
        return BuildBasePipeline(
            mlContext,
            //eventNameFeaturizedOutputColumnName: nameof(NormalizedFeatures.EventNameFeaturized),
            oneHotEncodedColumnNames: new (string output, string input)[]
            {
                (nameof(NormalizedFeatures.TheatreNameEncoded), nameof(CData.TheatreName)),
                (nameof(NormalizedFeatures.CityEncoded), nameof(CData.City)),
                (nameof(NormalizedFeatures.CountryEncoded), nameof(CData.Country)),
                (nameof(NormalizedFeatures.EventTypeEncoded), nameof(CData.EventType)),
                (nameof(NormalizedFeatures.SpokenLanguageEncoded), nameof(CData.SpokenLanguage)),
                (nameof(NormalizedFeatures.PresentationMethodEncoded), nameof(CData.PresentationMethod)),
            },
            eventGenresEncodedOutputColumnName: nameof(NormalizedFeatures.EventGenresEncoded),
            featureColumnNames: NormalizedFeatures.FeatureColumns(),
            logUsed: logUsed,
            normalizeColumnNames: _NormalizeMap
        );

    }

    /// <summary>
    /// Core builder for pipelines.
    /// </summary>
    /// <param name="mlContext">MLContext instance.</param>
    /// <param name="eventNameFeaturizedOutputColumnName">Name of the output column for featurized event names.</param>
    /// <param name="oneHotEncodedColumnNames">Array of input/output column name pairs for categorical encoding.</param>
    /// <param name="eventGenresEncodedOutputColumnName">Output column name for encoded genres.</param>
    /// <param name="featureColumnNames">List of all feature column names to concatenate into the final "Features" vector.</param>
    /// <param name="logUsed">If true, applies a logarithmic label transformation.</param>
    /// <param name="normalizeColumnNames">Optional normalization mapping (input/output) for numeric columns.</param>
    /// <returns>Data pipeline built according to the requirements.</returns>
    private static IEstimator<ITransformer> BuildBasePipeline(
        MLContext mlContext,
        //string eventNameFeaturizedOutputColumnName,
        (string output, string input)[] oneHotEncodedColumnNames,
        string eventGenresEncodedOutputColumnName,
        string[] featureColumnNames,
        bool logUsed,
        (string input, string output)[]? normalizeColumnNames = null)
    {
        mlContext.ComponentCatalog.RegisterAssembly(typeof(CyclicalTimeMapping).Assembly);
        mlContext.ComponentCatalog.RegisterAssembly(typeof(ExistenceFlagMapper).Assembly);
        mlContext.ComponentCatalog.RegisterAssembly(typeof(AverageAdmissionsMapper).Assembly);
        mlContext.ComponentCatalog.RegisterAssembly(typeof(RatingMapper).Assembly);
        mlContext.ComponentCatalog.RegisterAssembly(typeof(GenreMapper).Assembly);

        // Use the correct overload for CustomMapping
        var timeMap = mlContext.Transforms.CustomMapping<CData, CyclicalTime>(new CyclicalTimeMapping().GetMapping(), contractName: nameof(CyclicalTimeMapping));
        var existenceFlagMap = mlContext.Transforms.CustomMapping<CData, ExistenceFlags>(new ExistenceFlagMapper().GetMapping(), contractName: nameof(ExistenceFlagMapper));
        var averageAdmissionMap = mlContext.Transforms.CustomMapping<CData, AverageTimePeriodAdmissions>(new AverageAdmissionsMapper().GetMapping(), contractName: nameof(AverageAdmissionsMapper));
        var ratingMap = mlContext.Transforms.CustomMapping<CData, EventOrdinalRating>(new RatingMapper().GetMapping(), contractName: nameof(RatingMapper));


        IEstimator<ITransformer> pipeline =

            //mlContext.Transforms.Categorical.OneHotEncoding(eventNameFeaturizedOutputColumnName, nameof(CData.TheatreName))
            //mlContext.Transforms.Categorical.OneHotHashEncoding(eventNameFeaturizedOutputColumnName, nameof(CData.TheatreName))
            //mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "EventNameTokens", inputColumnName: nameof(CData.EventName))
            //.Append(mlContext.Transforms.Text.ApplyWordEmbedding(eventNameFeaturizedOutputColumnName, "EventNameTokens", modelKind: Microsoft.ML.Transforms.Text.WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D))
            //mlContext.Transforms.Text.FeaturizeText(eventNameFeaturizedOutputColumnName, nameof(CData.EventName))
            
            mlContext.Transforms.CustomMapping<CData, EventGenres>(new GenreMapper().GetMapping(), contractName: nameof(GenreMapper))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("EventGenresKeyed", nameof(EventGenres.EventGenresArray)))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(eventGenresEncodedOutputColumnName, "EventGenresKeyed", OneHotEncodingEstimator.OutputKind.Bag))

            .Append(timeMap)
            .Append(existenceFlagMap)
            .Append(averageAdmissionMap)
            .Append(ratingMap);

        foreach (var (output, input) in oneHotEncodedColumnNames)
        {
            pipeline = pipeline.Append(mlContext.Transforms.Categorical.OneHotEncoding(output, input));
        }

        if (normalizeColumnNames != null && normalizeColumnNames.Length > 0)
        {
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMinMax(
                normalizeColumnNames.Select(x => new InputOutputColumnPair(x.output, x.input)).ToArray()));
        }

        pipeline = pipeline
            // Keep a copy of the showtime identifying columns for later use
            .Append(mlContext.Transforms.CopyColumns("OriginalEventName", nameof(CData.EventName)))
            .Append(mlContext.Transforms.CopyColumns("OriginalEventName", nameof(CData.TheatreName)))
            .Append(mlContext.Transforms.CopyColumns("OriginalShowDateTime", nameof(CData.ShowDateTime)))
            .Append(mlContext.Transforms.Concatenate("Features", featureColumnNames));

        if (logUsed)
        {
            pipeline = pipeline.Append(mlContext.Transforms.CustomMapping(LogLabelMapper.Map, contractName: nameof(LogLabelMapper)));
        }

        return pipeline;
    }


}