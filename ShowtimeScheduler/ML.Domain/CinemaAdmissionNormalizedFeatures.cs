using Microsoft.ML.Data;

namespace ML.Domain;

/// <summary>
/// Defines the features that the machine learning model will be using.
/// </summary>
public class CinemaAdmissionNormalizedFeatures
{
    // Language Embedding
    //public float[] EventNameFeaturized { get; set; } = default!;
    public float[] AverageAdmissions { get; set; } = default!;
    public float[] TotalAverageAdmissions { get; set; } = default!;


    // Name of the Cinema Theatre - One-Hot Encode
    public float[] TheatreNameEncoded { get; set; } = default!;

    // Country Name - One-Hot Encode
    public float[] CountryEncoded { get; set; } = default!;

    // City Name - One-Hot Encode
    public float[] CityEncoded { get; set; } = default!;


    // Movie, Concert, LiveEvent - One-Hot Encode
    public float[] EventTypeEncoded { get; set; } = default!;

    // List of Genres separates by ", " - Multi-Hot Encode
    public float[] EventGenresEncoded { get; set; } = default!;

    // Custom Ordinal Encoding Based on the Minimum Age
    public float EventRatingEncodedNorm { get; set; } = default!;

    public float LengthInMinutesNorm { get; set; }


    // Estonian, Russian, English - One-Hot Encode
    public float[] SpokenLanguageEncoded { get; set; } = default!;

    // Regular or 3D - One-Hot Encode
    public float[] PresentationMethodEncoded { get; set; } = default!;

    // Weeks Since Release
    public float WeekNrNorm { get; set; }


    public float YearNorm { get; set; }

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

    public float PreViewAverageAdmissionsNorm { get; set; }

    public float PreViewAdmissionsNorm { get; set; }

    public float PreViewShowsNorm { get; set; }

    // Total Preview Data

    public float TotalPreViewShowsExist { get; set; }

    public float TotalPreViewAverageAdmissionsNorm { get; set; }

    public float TotalPreViewAdmissionsNorm { get; set; }

    public float TotalPreViewShowsNorm { get; set; }



    // First Week Data

    public float FirstWeekShowsExist { get; set; }

    public float FirstWeekAverageAdmissionsNorm { get; set; }

    public float FirstWeekAdmissionsNorm { get; set; }

    public float FirstWeekShowsNorm { get; set; }

    // Total First Week Data

    public float TotalFirstWeekShowsExist { get; set; }

    public float TotalFirstWeekAverageAdmissionsNorm { get; set; }

    public float TotalFirstWeekAdmissionsNorm { get; set; }

    public float TotalFirstWeekShowsNorm { get; set; }



    // Last Weekend Data

    public float LastWeekEndShowsExist { get; set; }

    public float LastWeekEndAverageAdmissionsNorm { get; set; }

    public float LastWeekEndAdmissionsNorm { get; set; }

    public float LastWeekEndShowsNorm { get; set; }



    // Last Week Data

    public float LastWeekExists { get; set; }

    public float LastWeekShowsExist { get; set; }

    public float LastWeekAverageAdmissionsNorm { get; set; }

    public float LastWeekAdmissionsNorm { get; set; }

    public float LastWeekShowsNorm { get; set; }

    // Total Last Week Data

    public float TotalLastWeekShowsExist { get; set; }

    public float TotalLastWeekAverageAdmissionsNorm { get; set; }

    public float TotalLastWeekAdmissionsNorm { get; set; }

    public float TotalLastWeekShowsNorm { get; set; }



    // PreLastWeek Data

    public float PreLastWeekExists { get; set; }

    public float PreLastWeekShowsExist { get; set; }

    public float PreLastWeekAverageAdmissionsNorm { get; set; }

    public float PreLastWeekAdmissionsNorm { get; set; }

    public float PreLastWeekShowsNorm { get; set; }

    // Total PreLastWeek Data

    public float TotalPreLastWeekShowsExist { get; set; }

    public float TotalPreLastWeekAverageAdmissionsNorm { get; set; }

    public float TotalPreLastWeekAdmissionsNorm { get; set; }

    public float TotalPreLastWeekShowsNorm { get; set; }



    // Label - Ticket Sales
    [ColumnName("Label")]
    public float Admissions { get; set; }


    public static string[] FeatureColumns()
    {
        return
        [
            //nameof(EventNameFeaturized),

            nameof(AverageAdmissions),
            nameof(TotalAverageAdmissions),

            nameof(TheatreNameEncoded),
            nameof(CountryEncoded),
            nameof(CityEncoded),

            nameof(EventTypeEncoded),
            nameof(EventGenresEncoded),
            nameof(EventRatingEncodedNorm),
            nameof(LengthInMinutesNorm),

            nameof(SpokenLanguageEncoded),
            nameof(PresentationMethodEncoded),

            nameof(WeekNrNorm),
            nameof(YearNorm),
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
            nameof(PreViewAdmissionsNorm),
            nameof(PreViewShowsNorm),
            nameof(PreViewAverageAdmissionsNorm),

            nameof(TotalPreViewShowsExist),
            nameof(TotalPreViewAdmissionsNorm),
            nameof(TotalPreViewShowsNorm),
            nameof(TotalPreViewAverageAdmissionsNorm),


            nameof(LastWeekExists), // acts as the first week exists, last weekend exists and last week exists flag


            nameof(FirstWeekShowsExist),
            nameof(FirstWeekAdmissionsNorm),
            nameof(FirstWeekShowsNorm),
            nameof(FirstWeekAverageAdmissionsNorm),

            nameof(TotalFirstWeekShowsExist),
            nameof(TotalFirstWeekAdmissionsNorm),
            nameof(TotalFirstWeekShowsNorm),
            nameof(TotalFirstWeekAverageAdmissionsNorm),


            nameof(LastWeekEndShowsExist),
            nameof(LastWeekEndAdmissionsNorm),
            nameof(LastWeekEndShowsNorm),
            nameof(LastWeekEndAverageAdmissionsNorm),


            nameof(LastWeekShowsExist),
            nameof(LastWeekAdmissionsNorm),
            nameof(LastWeekShowsNorm),
            nameof(LastWeekAverageAdmissionsNorm),

            nameof(TotalLastWeekShowsExist),
            nameof(TotalLastWeekAdmissionsNorm),
            nameof(TotalLastWeekShowsNorm),
            nameof(TotalLastWeekAverageAdmissionsNorm),



            nameof(PreLastWeekExists),
            nameof(PreLastWeekShowsExist),
            nameof(PreLastWeekAdmissionsNorm),
            nameof(PreLastWeekShowsNorm),
            nameof(PreLastWeekAverageAdmissionsNorm),

            nameof(TotalPreLastWeekShowsExist),
            nameof(TotalPreLastWeekAdmissionsNorm),
            nameof(TotalPreLastWeekShowsNorm),
            nameof(TotalPreLastWeekAverageAdmissionsNorm),

        ];
    }
}