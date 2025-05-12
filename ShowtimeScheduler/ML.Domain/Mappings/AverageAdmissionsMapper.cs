using Microsoft.ML.Transforms;
using ML.Domain.Features;

namespace ML.Domain.Mappings;

[CustomMappingFactoryAttribute(nameof(AverageAdmissionsMapper))]
public class AverageAdmissionsMapper : CustomMappingFactory<CinemaAdmissionData, AverageTimePeriodAdmissions>
{
    public override Action<CinemaAdmissionData, AverageTimePeriodAdmissions> GetMapping() 
    {
        return (input, output) =>
        {
            output.PreViewAverageAdmissions = input.PreViewShows > 0 ? (input.PreViewAdmissions / input.PreViewShows) : 0;
            output.TotalPreViewAverageAdmissions = input.TotalPreViewShows > 0 ? (input.TotalPreViewAdmissions / input.TotalPreViewShows) : 0;

            output.FirstWeekAverageAdmissions = input.FirstWeekShows > 0 ? (input.FirstWeekAdmissions / input.FirstWeekShows) : 0;
            output.TotalFirstWeekAverageAdmissions = input.TotalFirstWeekShows > 0 ? (input.TotalFirstWeekAdmissions / input.TotalFirstWeekShows) : 0;

            output.LastWeekEndAverageAdmissions = input.LastWeekEndShows > 0 ? (input.LastWeekEndAdmissions / input.LastWeekEndShows) : 0;

            output.LastWeekAverageAdmissions = input.LastWeekShows > 0 ? (input.LastWeekAdmissions / input.LastWeekShows) : 0;
            output.TotalLastWeekAverageAdmissions = input.TotalLastWeekShows > 0 ? (input.TotalLastWeekAdmissions / input.TotalLastWeekShows) : 0;

            output.PreLastWeekAverageAdmissions = input.PreLastWeekShows > 0 ? (input.PreLastWeekAdmissions / input.PreLastWeekShows) : 0;
            output.TotalPreLastWeekAverageAdmissions = input.TotalPreLastWeekShows > 0 ? (input.TotalPreLastWeekAdmissions / input.TotalPreLastWeekShows) : 0;
        };
    }
}