using Microsoft.ML.Data;

namespace ML.Domain;

public class PredictionWithContext
{
    public float Score { get; set; }
    public float Admissions { get; set; }

    public float Error => Math.Abs(Score - Admissions);

    public string TheatreName { get; set; } = default!;

    public string Country { get; set; } = default!;

    public string City { get; set; } = default!;


    public string EventName { get; set; } = default!;

    public string EventType { get; set; } = default!;

    public string EventGenres { get; set; } = default!;

    public string EventRating { get; set; } = default!;

    public float LengthInMinutes { get; set; }


    public string SpokenLanguage { get; set; } = default!;

    public string PresentationMethod { get; set; } = default!;

    public float WeekNr { get; set; }

    public DateTime ShowDateTime { get; set; }


    // Preview Data

    public float PreViewAdmissions { get; set; }

    public float PreViewShows { get; set; }

    // Total Preview Data

    public float TotalPreViewAdmissions { get; set; }

    public float TotalPreViewShows { get; set; }


    // First Week Data


    public float FirstWeekAdmissions { get; set; }

    public float FirstWeekShows { get; set; }

    // Total First Week Data

    public float TotalFirstWeekAdmissions { get; set; }

    public float TotalFirstWeekShows { get; set; }


    // Last Weekend Data

    public float LastWeekEndAdmissions { get; set; }

    public float LastWeekEndShows { get; set; }


    // Last Week Data

    public float LastWeekAdmissions { get; set; }

    public float LastWeekShows { get; set; }

    // Total Last Week Data

    public float TotalLastWeekAdmissions { get; set; }

    public float TotalLastWeekShows { get; set; }


    // PreLastWeek Data

    public float PreLastWeekAdmissions { get; set; }

    public float PreLastWeekShows { get; set; }

    // Total PreLastWeek Data

    public float TotalPreLastWeekAdmissions { get; set; }

    public float TotalPreLastWeekShows { get; set; }
}
