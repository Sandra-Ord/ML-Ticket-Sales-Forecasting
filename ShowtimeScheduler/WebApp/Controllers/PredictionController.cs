using App.DTO;
using Microsoft.AspNetCore.Mvc;
using ML.Domain;
using ML.Services;

namespace WebApp.Controllers;

/// <summary>
/// Controller for handling prediction-related requests using trained ML models.
/// </summary>
[ApiController]
[Route("api/Prediction")]
public class PredictionController : ControllerBase
{
    //private readonly ModelProvider _modelProvider;
    private readonly InferenceEngine _engine;

    /// <summary>
    /// Initializes a new instance of the <see cref="PredictionController"/> class.
    /// </summary>
    /// <param name="engine">Inference engine used to make predictions.</param>
    public PredictionController(InferenceEngine engine)
    {
        _engine = engine;
    }

    /// <summary>
    /// Predicts attendance for a single movie showtime.
    /// </summary>
    /// <param name="input">Showtime data.</param>
    /// <returns></returns>
    [HttpPost("Admissions")]
    public IActionResult PredictAdmissions([FromBody] CinemaAdmissionData input)
    {
        if (input == null)
        {
            return BadRequest(new PredictionResult<PredictionWithMetadata>
            {
                Errors = { "Request body was null." }
            });
        }

        try
        {
            var result = _engine.PredictAdmission(input);
            if (!result.IsSuccess)
            {
                return BadRequest(result);
            }
            return Ok(result);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error during prediction: {ex.Message}");
        }
    }

    /// <summary>
    /// Predicts attendance for a batch of movie showtimes.
    /// </summary>
    /// <param name="input">List of showtimes.</param>
    /// <returns></returns>
    [HttpPost("BatchAdmissions")]
    public IActionResult BatchPredictAdmissions([FromBody] List<CinemaAdmissionData> input)
    {
        if (input == null )
        {
            return BadRequest(new PredictionResult<PredictionWithMetadata>
            {
                Errors = { "Request body was null." }
            });
        }

        try
        {
            var result = _engine.BatchPredictAdmission(input);
            return Ok(result);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error during prediction: {ex.Message}");
        }
    }

    /// <summary>
    /// Predicts attendance of a showtime for specific slot times.
    /// </summary>
    /// <param name="request">Showtime details and list of time slots for attendance predicting.</param>
    /// <returns></returns>
    [HttpPost("BatchSlotAdmissions")]
    public IActionResult BatchPredictSlotAdmissions([FromBody] BatchSlotAdmissionsRequest request)
    {
        if (request.SlotTimes == null)
        {
            return BadRequest(new PredictionResult<PredictionWithMetadata>
            {
                Errors = { "Request body was null." }
            });
        }

        try
        {
            var result = _engine.BatchPredictSlotAdmissions(request.Input, request.SlotTimes);
            return Ok(result);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error during prediction: {ex.Message}");
        }
    }

    /// <summary>
    /// Predicts the highest attendance showtime within the given time window.
    /// </summary>
    /// <param name="request">Showtime details and the start and end of the time window, where the highest attendance is to be found.</param>
    /// <returns></returns>
    [HttpPost("BestShowTime")]
    public IActionResult PredictBestShowTime([FromBody] BestShowTimeRequest request)
    {
        if (request.Input == null)
        {
            return BadRequest(new PredictionResult<PredictionWithMetadata>
            {
                Errors = { "Request body was null." }
            });
        }

        try
        {
            var result = _engine.PredictBestShowTime(request.Input, request.EarliestStartTime == null ? TimeSpan.FromHours(request.EarliestStartTime.Value) : null, request.EarliestStartTime == null ? TimeSpan.FromHours(request.LatestStartTime.Value) : null);
            if (!result.IsSuccess)
            {
                return BadRequest(result);
            }
            return Ok(result);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error during prediction: {ex.Message}");
        }
    }

    /// <summary>
    /// Predicts the top N highest attendance showtimes within the given time window.
    /// </summary>
    /// <param name="request">Showtime details, number of top results and the start and end of the time window, where the highest attendance is to be found.</param>
    /// <returns></returns>
    [HttpPost("TopNBestShowTime")]
    public IActionResult PredictTopNBestShowTime([FromBody] TopNBestShowTimesRequest request)
    {
        if (request.Input == null)
        {
            return BadRequest(new PredictionResult<PredictionWithMetadata>
            {
                Errors = { "Request body was null." }
            });
        }

        try
        {
            var result = _engine.PredictTopNBestShowTime(request.Input, request.TopN, request.EarliestStartTime == null ? TimeSpan.FromHours(request.EarliestStartTime.Value) : null, request.EarliestStartTime == null ? TimeSpan.FromHours(request.LatestStartTime.Value) : null);
            if (!result.IsSuccess)
            {
                return BadRequest(result);
            }
            return Ok(result);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error during prediction: {ex.Message}");
        }
    }



    [HttpPost("PredictFill")]
    public IActionResult Fill([FromBody] string input)
    {
        var result = 22;

        return Ok(result);
    }
    
}