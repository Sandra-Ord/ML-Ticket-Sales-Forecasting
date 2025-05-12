using Microsoft.ML;
using ML.Domain;

namespace ML.Data;

public static class DataProvider
{
    public static IDataView LoadFromFile(MLContext mlContext, string fileName, char separator = ';', bool header = true)
    {
        return mlContext.Data.LoadFromTextFile<CinemaAdmissionData>(fileName, separatorChar: separator, hasHeader: header);
    }

    public static IDataView LoadFromDataBase(MLContext mlContext)
    {
        return mlContext.Data.LoadFromTextFile<CinemaAdmissionData>("", separatorChar: ';', hasHeader: true);
    }

    public static IDataView FilterByColumn(MLContext mlContext, IDataView data, string columnName, double lower = double.NegativeInfinity, double upper = double.PositiveInfinity)
    {
        return mlContext.Data.FilterRowsByColumn(data, columnName, lowerBound: lower, upperBound: upper);

    }

}
