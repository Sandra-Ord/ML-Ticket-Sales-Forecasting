using ML.Domain;

namespace App.DTO;

/// <summary>
/// Represents a request to get the top N best showtimes for a movie within the window.
/// </summary>
public class TopNBestShowTimesRequest
{
    /// <summary>
    /// Input data representing all of the showtime details (the date of the ShowTimeStart is important, the time element is not).
    /// </summary>
    public CinemaAdmissionData Input { get; set; } = default!;

    /// <summary>
    /// Number of top showtimes to return.
    /// </summary>
    public int TopN { get; set; } = 3;

    /// <summary>
    /// Assess the showtimes starting from the earliest start time.
    /// </summary>
    public double? EarliestStartTime { get; set; }

    /// <summary>
    /// Assess the showtimes ending at the latest start time.
    /// </summary>
    public double? LatestStartTime { get; set; }
}