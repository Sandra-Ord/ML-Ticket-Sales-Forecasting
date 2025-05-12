using Microsoft.ML.Transforms;
using ML.Domain.Features;
using Utils;

namespace ML.Domain.Mappings;

[CustomMappingFactoryAttribute(nameof(CyclicalTimeMapping))]
public class CyclicalTimeMapping : CustomMappingFactory<CinemaAdmissionData, CyclicalTime>
{
    public override Action<CinemaAdmissionData, CyclicalTime> GetMapping()
    {
        return (input, output) =>
        {
            var dt = input.ShowDateTime;

            output.Year = dt.Year;

            output.IsWeekEnd = (dt.DayOfWeek == DayOfWeek.Friday ||
                                dt.DayOfWeek == DayOfWeek.Saturday ||
                                dt.DayOfWeek == DayOfWeek.Sunday) ? 1 : 0;

            (output.SinMonth, output.CosMonth) = CyclicalHelper.CalculateCycle(dt.Month - 1, 12);

            (output.SinDayOfMonth, output.CosDayOfMonth) = CyclicalHelper.CalculateCycle(dt.Day - 1, DateTime.DaysInMonth(dt.Year, dt.Month));

            (output.SinDayOfYear, output.CosDayOfYear) = CyclicalHelper.CalculateCycle(dt.DayOfYear - 1, DateTime.IsLeapYear(dt.Year) ? 366 : 365);

            (output.SinWeekDay, output.CosWeekDay) = CyclicalHelper.CalculateCycle((int)dt.DayOfWeek, 7);

            (output.SinHour, output.CosHour) = CyclicalHelper.CalculateCycle(dt.Hour, 24);

            (output.SinMinuteOfDay, output.CosMinuteOfDay) = CyclicalHelper.CalculateCycle(dt.Hour * 60 + dt.Minute, 1440);
        };
    }
}