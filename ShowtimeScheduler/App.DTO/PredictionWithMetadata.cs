namespace App.DTO;

/// <summary>
/// Represents a prediction result, including metadata about the event and starting time of the showtime in order to ensure correct interpretation of the results.
/// </summary>
public record PredictionWithMetadata
{
    /// <summary>
    /// Name of the event (e.g. movie title).
    /// </summary>
    public string EventName { get; set; } = default!;

    /// <summary>
    /// Date and time of the start of the show for which the prediction was made.
    /// </summary>
    public DateTime ShowDateTime { get; set; } = default;

    /// <summary>
    /// Predicted attendance for the specific event and showtime.
    /// </summary>
    public float Attendance { get; set; }

    public PredictionWithMetadata(string eventName, DateTime showDateTime, float attendance)
    {
        EventName = eventName;
        ShowDateTime = showDateTime;
        Attendance = attendance;
    }
}