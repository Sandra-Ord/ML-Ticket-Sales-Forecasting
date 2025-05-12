using Microsoft.ML.Transforms;
using ML.Domain.Features;

namespace ML.Domain.Mappings;

[CustomMappingFactoryAttribute(nameof(ExistenceFlagMapper))]
public class ExistenceFlagMapper : CustomMappingFactory<CinemaAdmissionData, ExistenceFlags>
{
    public override Action<CinemaAdmissionData, ExistenceFlags> GetMapping()
    {
        return (input, output) =>
        {
            var preViewExists = input.WeekNr >= 1;
            output.PreViewExists = preViewExists ? 1 : 0;
            output.PreViewShowsExist = preViewExists ? (input.PreViewShows > 0 ? 1 : 0) : 0;
            output.TotalPreViewShowsExist = preViewExists ? (input.TotalPreViewShows > 0 ? 1 : 0) : 0;

            var lastWeekExists = input.WeekNr >= 2;


            output.FirstWeekShowsExist = lastWeekExists ? (input.FirstWeekShows > 0 ? 1 : 0) : 0;
            output.TotalFirstWeekShowsExist = lastWeekExists ? (input.TotalFirstWeekShows > 0 ? 1 : 0) : 0;

            output.LastWeekEndShowsExist = lastWeekExists ? (input.LastWeekEndShows > 0 ? 1 : 0) : 0;

            output.LastWeekExists = lastWeekExists ? 1 : 0;
            output.LastWeekShowsExist = lastWeekExists ? (input.LastWeekShows > 0 ? 1 : 0) : 0;
            output.TotalLastWeekShowsExist = lastWeekExists ? (input.TotalLastWeekShows > 0 ? 1 : 0) : 0;

            var preLastWeekExists = input.WeekNr >= 3;
            output.PreLastWeekExists = preLastWeekExists ? 1 : 0;
            output.PreLastWeekShowsExist = preLastWeekExists ? (input.PreLastWeekShows > 0 ? 1 : 0) : 0;
            output.TotalPreLastWeekShowsExist = preLastWeekExists ? (input.TotalPreLastWeekShows > 0 ? 1 : 0) : 0;
        };
    }
}