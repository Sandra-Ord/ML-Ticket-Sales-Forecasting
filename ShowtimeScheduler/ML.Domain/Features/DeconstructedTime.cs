namespace ML.Domain.Features;

public class DeconstructedTime
{
    public float Year { get; set; }

    public float Month { get; set; }

    public float DayOfMonth { get; set; }

    public float DayOfYear { get; set; }

    public float WeekDay { get; set; }

    public float IsWeekEnd { get; set; }

    public float Hour { get; set; }

    public float MinuteOfDay { get; set; }
}