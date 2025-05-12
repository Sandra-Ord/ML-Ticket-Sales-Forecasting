using ML.Domain.Features;

namespace ML.Domain.Mappings;

public class WeightMapper
{
    public static Action<CinemaAdmissionData, DataWeight> Map = (input, output) =>
    {
        if (input.Label <= 10) output.Weight = 1f;
        else if (input.Label <= 30) output.Weight = 2f;
        else if (input.Label <= 75) output.Weight = 5f;
        else if (input.Label <= 100) output.Weight = 1f;
        else output.Weight = 1.5f;
    };
}