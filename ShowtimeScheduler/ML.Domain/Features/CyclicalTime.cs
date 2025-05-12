namespace ML.Domain.Features;

public class CyclicalTime
{
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

}
