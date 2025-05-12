using App.DTO;
using Microsoft.Extensions.ML;
using Microsoft.ML;
using ML.Domain;
using ML.Domain.Features;

namespace ML.Services;

/// <summary>
/// Provides methods for performing inference using a trainer ML.NET prediction model.
/// </summary>
public class InferenceEngine
{

    private readonly PredictionEnginePool<CinemaAdmissionData, ScoreOnly> _pool;

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceEngine"/> class with a prediction engine pool.
    /// </summary>
    /// <param name="pool">Prediction engine pool for making inferences.</param>
    public InferenceEngine(PredictionEnginePool<CinemaAdmissionData, ScoreOnly> pool)
    {
        _pool = pool;
    }

    /// <summary>
    /// Predicts attendance based on a single input sample.
    /// </summary>
    /// <param name="input">Input data for the prediction.</param>
    /// <returns>Prediction result including relevant metadata (title of the movie, DateTime of the showtime, predicted attendance).</returns>
    public PredictionWithMetadata Predict(CinemaAdmissionData input)
    {
        // TODO should the metadata also include the cinema? probably should
        var scoreOnly = _pool.Predict("AdmissionModel", input);
        return new PredictionWithMetadata(
            input.EventName,
            input.ShowDateTime,
            scoreOnly.Score);
    }

    /// <summary>
    /// Predicts attendance with metadata and includes warnings based on the input.
    /// </summary>
    /// <param name="input">Input data for prediction.</param>
    /// <returns>Structured prediction result containing the predicted value with relevant metadata (to ensure that the results are interpreted correctly) and any warnings or errors.</returns>
    public PredictionResult<PredictionWithMetadata> PredictAdmission(CinemaAdmissionData input)
    {
        var result = Predict(input);
        var predictionResult = new PredictionResult<PredictionWithMetadata>
        {
            Value = result
        };
        if (input.WeekNr <= 1)
        {
            predictionResult.Warnings.Add("The movie is very new (week number <= 1). The prediction may not be accurate.");
        }
        return predictionResult;
    }

    /// <summary>
    /// Predicts attendance for a batch of showtimes.
    /// </summary>
    /// <param name="input">List of showtime data points.</param>
    /// <returns>List of prediction results for each showtime (each result contains the prediction and relevant metadata and any warnings and errors.)</returns>
    public List<PredictionResult<PredictionWithMetadata>> BatchPredictAdmission(List<CinemaAdmissionData> input)
    {
        var result = new List<PredictionResult<PredictionWithMetadata>>();
        foreach (var showtime in input)
        {
            result.Add(PredictAdmission(showtime));
        }
        return result;
    }

    /// <summary>
    /// Predicts the attendance for the showtime data point at all the different time slots.
    /// </summary>
    /// <param name="baseData">Showtime data.</param>
    /// <param name="showTimes">List of different DateTime values to predict the attendance for.</param>
    /// <returns>List of prediction results for each showtime (each result contains the prediction and relevant metadata and any warnings and errors.)</returns>
    public List<PredictionResult<PredictionWithMetadata>> BatchPredictSlotAdmissions(
        CinemaAdmissionData baseData,
        IEnumerable<DateTime> showTimes)
    {
        var results = new List<PredictionResult<PredictionWithMetadata>>();

        foreach (var dt in showTimes)
        {
            var candidate = baseData with { ShowDateTime = dt };

            var slotResult = PredictAdmission(candidate);

            results.Add(slotResult);
        }

        return results;
    }

    /// <summary>
    /// Predicts the highest attendance showtime within the time window.
    /// 
    /// If no time window is set, the highest attendance will be predicted within the time window of general opening and closing hours of the cinema (10:00 - 23:00).
    /// </summary>
    /// <param name="input">Showtime data.</param>
    /// <param name="earliestStartTime">Optional earliest time the show can start (defaults to 10:15 - the cinema opens 15 minutes before the first showtime starts but no earlier than 10:00).</param>
    /// <param name="latestStartTime">Optional latest time the show can start (default to 22:45 - the cinema closes 15 after the last showtime of the day has started).</param>
    /// <returns>Structured prediction result containing the predicted value with relevant metadata (to ensure that the results are interpreted correctly) and any warnings or errors.</returns>
    public PredictionResult<PredictionWithMetadata> PredictBestShowTime(
        CinemaAdmissionData input, 
        TimeSpan? earliestStartTime = null, 
        TimeSpan? latestStartTime = null)
    {
        var result = PredictTopNBestShowTime(input, 1, earliestStartTime, latestStartTime);
        return new PredictionResult<PredictionWithMetadata>()
        {
            Value = result.Value?.FirstOrDefault(),
            Errors = result.Errors,
            Warnings = result.Warnings
        };
        
    }

    /// <summary>
    /// Predicts the top N highest attendance showtimes within the time window.
    /// 
    /// If no time window is set, the highest attendance will be predicted within the time window of general opening and closing hours of the cinema (10:00 - 23:00).
    /// </summary>
    /// <param name="input">Showtime data.</param>
    /// <param name="topN">Number of top results to return.</param>
    /// <param name="earliestStartTime">Optional earliest time the show can start (defaults to 10:15 - the cinema opens 15 minutes before the first showtime starts but no earlier than 10:00).</param>
    /// <param name="latestStartTime">Optional latest time the show can start (default to 22:45 - the cinema closes 15 after the last showtime of the day has started).</param>
    /// <returns>List of prediction results for each showtime (each result contains the prediction and relevant metadata and any warnings and errors.)</returns>
    public PredictionResult<IEnumerable<PredictionWithMetadata>> PredictTopNBestShowTime(
    CinemaAdmissionData input,
    int topN = 3,
    TimeSpan? earliestStartTime = null,
    TimeSpan? latestStartTime = null)
    {
        // The cinema opens 15 minutes before the earliest showtime [, but not earlier than 10:00]
        earliestStartTime ??= TimeSpan.FromHours(10.25); // 10:15
        // The cinema closes 15 minutes after the latest showtime [, but not later than 23:00]
        latestStartTime ??= TimeSpan.FromHours(22.75);   // 22:45

        // The cinema showtimes always start at round time values
        var interval = TimeSpan.FromMinutes(5);



        if (earliestStartTime > latestStartTime)
        {
            return new PredictionResult<IEnumerable<PredictionWithMetadata>>
            {
                Errors = { "Invalid time window: earliest start must be before latest start." }
            };
        }

        var predictionResult = new PredictionResult<IEnumerable<PredictionWithMetadata>>();

        int slots = (int)((latestStartTime.Value - earliestStartTime.Value) / interval) + 1;
        if (slots < topN)
        {
            predictionResult.Warnings.Add($"Only {slots} predictions could be made within the specified time window, which is fewer than the requested top {topN}.");
        }


        if (input.WeekNr <= 1)
        {
            predictionResult.Warnings.Add("The movie is very new (week number <= 1). The prediction may be less accurate.");
        }

        var allPredictions = EvaluateAllShowTimes(input, earliestStartTime.Value, latestStartTime.Value, interval).Take(topN);

        predictionResult.Value = allPredictions;

        return predictionResult;
    }

    private List<PredictionWithMetadata> EvaluateAllShowTimes(
        CinemaAdmissionData input,
        TimeSpan earliestStartTime,
        TimeSpan latestStartTime,
        TimeSpan interval)
    {
        var testInput = input;
        var baseDate = testInput.ShowDateTime.Date;

        var results = new List<PredictionWithMetadata>();

        for (var time = earliestStartTime; time <= latestStartTime; time += interval)
        {
            testInput.ShowDateTime = baseDate + time;
            results.Add(Predict(testInput));
        }

        return results.OrderByDescending(p => p.Attendance).ToList();
    }
}
