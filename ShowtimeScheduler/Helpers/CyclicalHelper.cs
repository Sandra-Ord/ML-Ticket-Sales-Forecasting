namespace Utils;

public class CyclicalHelper
{
    public static float SinCycle(double position, double period)
    {
        return (float)Math.Sin(2 * Math.PI * position / period);
    }

    public static float CosCycle(double position, double period)
    {
        return (float)Math.Cos(2 * Math.PI * position / period);
    }

    public static (float sin, float cos) CalculateCycle(float value, float period)
    {
        return (SinCycle(value, period), CosCycle(value, period));
    }

}
