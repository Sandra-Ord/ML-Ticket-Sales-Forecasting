using Microsoft.ML.Data;

namespace ML.Domain;

/// <summary>
/// Defines the features that the machine learning uses.
/// </summary>
public class CinemaAdmissionFeatures
{
    // Name of the Cinema Theatre - One-Hot Encode
    public float[] TheatreNameEncoded { get; set; } = default!;

    // Country Name - One-Hot Encode
    public float[] CountryEncoded { get; set; } = default!;

    // City Name - One-Hot Encode
    public float[] CityEncoded { get; set; } = default!;


    // Language Embedding
    //public float[] EventNameFeaturized { get; set; } = default!;

    public float AverageAdmissions { get; set; } = default!;

    public float TotalAverageAdmissions { get; set; } = default!;


    // Movie, Concert, LiveEvent - One-Hot Encode
    public float[] EventTypeEncoded { get; set; } = default!;

    // List of Genres separates by ", " - Multi-Hot Encode
    public float[] EventGenresEncoded { get; set; } = default!;

    // Custom Ordinal Encoding Based on the Minimum Age
    public float EventRatingEncoded { get; set; } = default!;

    public float LengthInMinutes { get; set; }


    // Estonian, Russian, English - One-Hot Encode
    public float[] SpokenLanguageEncoded { get; set; } = default!;

    // Regular or 3D - One-Hot Encode
    public float[] PresentationMethodEncoded { get; set; } = default!;

    // Weeks Since Release
    public float WeekNr { get; set; }


    public float Year { get; set; }

    public float IsWeekEnd { get; set; }



    // Cyclical Date Values

    public float SinMonth { get; set; } = default!;
    public float CosMonth { get; set; } = default!;

    public float SinDayOfMonth { get; set; } = default!;
    public float CosDayOfMonth { get; set; } = default!;

    public float SinDayOfYear { get; set; } = default!;
    public float CosDayOfYear { get; set; } = default!;

    public float SinWeekDay { get; set; } = default!;
    public float CosWeekDay { get; set; } = default!;

    public float SinHour { get; set; } = default!;
    public float CosHour { get; set; } = default!;

    public float SinMinuteOfDay { get; set; } = default!;
    public float CosMinuteOfDay { get; set; } = default!;



    // Preview Data

    public float PreViewExists { get; set; }

    public float PreViewShowsExist { get; set; }

    public float PreViewAverageAdmissions { get; set; }

    public float PreViewAdmissions { get; set; }

    public float PreViewShows { get; set; }

    // Total Preview Data

    public float TotalPreViewShowsExist { get; set; }

    public float TotalPreViewAverageAdmissions { get; set; }

    public float TotalPreViewAdmissions { get; set; }

    public float TotalPreViewShows { get; set; }



    // First Week Data

    public float FirstWeekShowsExist { get; set; }

    public float FirstWeekAverageAdmissions { get; set; }

    public float FirstWeekAdmissions { get; set; }

    public float FirstWeekShows { get; set; }

    // Total First Week Data

    public float TotalFirstWeekShowsExist { get; set; }

    public float TotalFirstWeekAverageAdmissions { get; set; }

    public float TotalFirstWeekAdmissions { get; set; }

    public float TotalFirstWeekShows { get; set; }



    // Last Weekend Data

    public float LastWeekEndShowsExist { get; set; }

    public float LastWeekEndAverageAdmissions { get; set; }

    public float LastWeekEndAdmissions { get; set; }

    public float LastWeekEndShows { get; set; }



    // Last Week Data

    public float LastWeekExists { get; set; }

    public float LastWeekShowsExist { get; set; }

    public float LastWeekAverageAdmissions { get; set; }

    public float LastWeekAdmissions { get; set; }

    public float LastWeekShows { get; set; }

    // Total Last Week Data

    public float TotalLastWeekShowsExist { get; set; }

    public float TotalLastWeekAverageAdmissions { get; set; }

    public float TotalLastWeekAdmissions { get; set; }

    public float TotalLastWeekShows { get; set; }



    // PreLastWeek Data

    public float PreLastWeekExists { get; set; }

    public float PreLastWeekShowsExist { get; set; }

    public float PreLastWeekAverageAdmissions { get; set; }

    public float PreLastWeekAdmissions { get; set; }

    public float PreLastWeekShows { get; set; }

    // Total PreLastWeek Data

    public float TotalPreLastWeekShowsExist { get; set; }

    public float TotalPreLastWeekAverageAdmissions { get; set; }

    public float TotalPreLastWeekAdmissions { get; set; }

    public float TotalPreLastWeekShows { get; set; }



    // Label - Ticket Sales
    [ColumnName("Label")]
    public float Admissions { get; set; }

    public static string[] OriginalFeatureColumns()
    {
        // Commented out features were included in the original training but removed after the PFI analysis
        return
        [
            nameof(AverageAdmissions),
            nameof(TotalAverageAdmissions),

            nameof(TheatreNameEncoded),
            nameof(CountryEncoded),
            nameof(CityEncoded),

            nameof(EventTypeEncoded),
            nameof(EventGenresEncoded),
            nameof(EventRatingEncoded),
            nameof(LengthInMinutes),

            nameof(SpokenLanguageEncoded),
            nameof(PresentationMethodEncoded),

            nameof(WeekNr),
            nameof(Year),
            nameof(IsWeekEnd),

            nameof(SinMonth),
            nameof(CosMonth),
            nameof(SinDayOfMonth),
            nameof(CosDayOfMonth),
            nameof(SinDayOfYear),
            nameof(CosDayOfYear),
            nameof(SinWeekDay),
            nameof(CosWeekDay),
            nameof(SinHour),
            nameof(CosHour),
            nameof(SinMinuteOfDay),
            nameof(CosMinuteOfDay),


            nameof(PreViewExists),
            nameof(PreViewShowsExist),
            nameof(PreViewAdmissions),
            nameof(PreViewShows),
            nameof(PreViewAverageAdmissions),

            nameof(TotalPreViewShowsExist),
            nameof(TotalPreViewAdmissions),
            nameof(TotalPreViewShows),
            nameof(TotalPreViewAverageAdmissions),

            nameof(LastWeekExists), 


            nameof(FirstWeekShowsExist),
            nameof(FirstWeekAdmissions),
            nameof(FirstWeekShows),
            nameof(FirstWeekAverageAdmissions),

            nameof(TotalFirstWeekShowsExist),
            nameof(TotalFirstWeekAdmissions),
            nameof(TotalFirstWeekShows),
            nameof(TotalFirstWeekAverageAdmissions),


            nameof(LastWeekEndShowsExist),
            nameof(LastWeekEndAdmissions),
            nameof(LastWeekEndShows),
            nameof(LastWeekEndAverageAdmissions),


            nameof(LastWeekShowsExist),
            nameof(LastWeekAdmissions),
            nameof(LastWeekShows),
            nameof(LastWeekAverageAdmissions),

            nameof(TotalLastWeekShowsExist),
            nameof(TotalLastWeekAdmissions),
            nameof(TotalLastWeekShows),
            nameof(TotalLastWeekAverageAdmissions),


            nameof(PreLastWeekExists),
            nameof(PreLastWeekShowsExist),
            nameof(PreLastWeekAdmissions),
            nameof(PreLastWeekShows),
            nameof(PreLastWeekAverageAdmissions),

            nameof(TotalPreLastWeekShowsExist),  
            nameof(TotalPreLastWeekAdmissions),
            nameof(TotalPreLastWeekShows),
            nameof(TotalPreLastWeekAverageAdmissions),
        ];
    }

    public static string[] FeatureColumns()
    {
        // Commented out features were included in the original training but removed after the PFI analysis
        return
        [
            nameof(AverageAdmissions),
            nameof(TotalAverageAdmissions),

            nameof(TheatreNameEncoded),
            nameof(CountryEncoded),
            nameof(CityEncoded),

            //nameof(EventTypeEncoded),
            nameof(EventGenresEncoded),
            nameof(EventRatingEncoded),
            nameof(LengthInMinutes),

            nameof(SpokenLanguageEncoded),
            nameof(PresentationMethodEncoded), 

            nameof(WeekNr),
            nameof(Year),
            nameof(IsWeekEnd),

            nameof(SinMonth),
            nameof(CosMonth),
            nameof(SinDayOfMonth), 
            nameof(CosDayOfMonth), 
            nameof(SinDayOfYear),
            nameof(CosDayOfYear),
            nameof(SinWeekDay),
            nameof(CosWeekDay),
            nameof(SinHour),
            nameof(CosHour),
            nameof(SinMinuteOfDay), 
            nameof(CosMinuteOfDay), 


            //nameof(PreViewExists),
            //nameof(PreViewShowsExist),
            nameof(PreViewAdmissions),
            nameof(PreViewShows),
            nameof(PreViewAverageAdmissions),

            //nameof(TotalPreViewShowsExist),
            nameof(TotalPreViewAdmissions),
            //nameof(TotalPreViewShows),
            nameof(TotalPreViewAverageAdmissions),

            //nameof(LastWeekExists), 


            //nameof(FirstWeekShowsExist),
            nameof(FirstWeekAdmissions),
            nameof(FirstWeekShows),
            nameof(FirstWeekAverageAdmissions),

            //nameof(TotalFirstWeekShowsExist),
            nameof(TotalFirstWeekAdmissions),
            nameof(TotalFirstWeekShows), 
            nameof(TotalFirstWeekAverageAdmissions),


            //nameof(LastWeekEndShowsExist),
            nameof(LastWeekEndAdmissions),
            nameof(LastWeekEndShows), 
            nameof(LastWeekEndAverageAdmissions),


            //nameof(LastWeekShowsExist),
            nameof(LastWeekAdmissions),
            nameof(LastWeekShows),
            nameof(LastWeekAverageAdmissions),

            //nameof(TotalLastWeekShowsExist),
            nameof(TotalLastWeekAdmissions),
            nameof(TotalLastWeekShows),
            nameof(TotalLastWeekAverageAdmissions),


            //nameof(PreLastWeekExists),
            //nameof(PreLastWeekShowsExist), 
            nameof(PreLastWeekAdmissions),
            nameof(PreLastWeekShows), 
            nameof(PreLastWeekAverageAdmissions),

            //nameof(TotalPreLastWeekShowsExist),  
            nameof(TotalPreLastWeekAdmissions),
            //nameof(TotalPreLastWeekShows),
            nameof(TotalPreLastWeekAverageAdmissions),
        ];
    }

}