using Microsoft.ML.Data;

namespace ML.Domain;

/// <summary>
///  Data defines the data structure that can be fetched directly from the source.
/// </summary>
public record CinemaAdmissionData
{

    // Name of the Cinema Theatre - One-Hot Encode
    [LoadColumn(0)]
    public string TheatreName { get; set; } = default!;

    // Country Name - One-Hot Encode
    [LoadColumn(1)]
    public string Country { get; set; } = default!;

    // City Name - One-Hot Encode
    [LoadColumn(2)]
    public string City { get; set; } = default!;


    // Language Embedding
    [LoadColumn(3)]
    public string EventName { get; set; } = default!;

    [LoadColumn(4)]
    public float AverageAdmissions { get; set; } = default!;

    [LoadColumn(5)]
    public float TotalAverageAdmissions { get; set; } = default!;


    // Movie, Concert, LiveEvent - One-Hot Encode
    [LoadColumn(6)]
    public string EventType { get; set; } = default!;

    // List of Genres separates by ", " - Multi-Hot Encode
    [LoadColumn(7)]
    public string EventGenres { get; set; } = default!;

    // Custom Ordinal Encoding Based on the Minimum Age
    [LoadColumn(8)]
    public string EventRating { get; set; } = default!;

    [LoadColumn(9)]
    public float LengthInMinutes { get; set; }


    // Estonian, Russian, English - One-Hot Encode
    [LoadColumn(10)]
    public string SpokenLanguage { get; set; } = default!;

    // Regular or 3D - One-Hot Encode
    [LoadColumn(11)]
    public string PresentationMethod { get; set; } = default!;

    // Weeks Since Release
    [LoadColumn(12)]
    public float WeekNr { get; set; }

    // Date and Time of the Start of the ShowTime
    [LoadColumn(13)]
    public DateTime ShowDateTime { get; set; }


    // Preview Data

    [LoadColumn(14)]
    public float PreViewAdmissions { get; set; }

    [LoadColumn(15)]
    public float PreViewShows { get; set; }

    // Total Preview Data

    [LoadColumn(16)]
    public float TotalPreViewAdmissions { get; set; }

    [LoadColumn(17)]
    public float TotalPreViewShows { get; set; }


    // First Week Data


    [LoadColumn(18)] 
    public float FirstWeekAdmissions { get; set; }

    [LoadColumn(19)]
    public float FirstWeekShows { get; set; }

    // Total First Week Data

    [LoadColumn(20)]
    public float TotalFirstWeekAdmissions { get; set; }

    [LoadColumn(21)]
    public float TotalFirstWeekShows { get; set; }


    // Last Weekend Data

    [LoadColumn(22)]
    public float LastWeekEndAdmissions { get; set; }

    [LoadColumn(23)]
    public float LastWeekEndShows { get; set; }


    // Last Week Data

    [LoadColumn(24)]
    public float LastWeekAdmissions { get; set; }

    [LoadColumn(25)]
    public float LastWeekShows { get; set; }

    // Total Last Week Data

    [LoadColumn(26)]
    public float TotalLastWeekAdmissions { get; set; }

    [LoadColumn(27)]
    public float TotalLastWeekShows { get; set; }


    // PreLastWeek Data

    [LoadColumn(28)]
    public float PreLastWeekAdmissions { get; set; }

    [LoadColumn(29)]
    public float PreLastWeekShows { get; set; }

    // Total PreLastWeek Data

    [LoadColumn(30)]
    public float TotalPreLastWeekAdmissions { get; set; }

    [LoadColumn(31)]
    public float TotalPreLastWeekShows { get; set; }


    // Label - Ticket Sales

    [LoadColumn(32)]
    [ColumnName("Label")]
    public float Label { get; set; }

}
