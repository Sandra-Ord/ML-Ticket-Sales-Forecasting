using ML.Domain.Features;

namespace ML.Domain.Mappings;

public static class LogLabelMapper
{
    public static Action<CinemaAdmissionData, AdmissionLabel> Map = (input, output) =>
    {
        var capped = Math.Min(input.Label, 300);
        output.Label = (float)Math.Log(capped + 1);
    };
}