using ML.Domain;

namespace App.DTO;

/// <summary>
/// Represents a request to evaluate multiple showtimes for a movie.
/// </summary>
public class BatchSlotAdmissionsRequest
{
    /// <summary>
    /// Input data representing all of the showtime details (the date of the showtime is not important).
    /// </summary>
    public CinemaAdmissionData Input { get; set; } = default!;

    /// <summary>
    /// List of proposed slot start times to evaluate.
    /// </summary>
    public List<DateTime> SlotTimes { get; set; } = new();
}