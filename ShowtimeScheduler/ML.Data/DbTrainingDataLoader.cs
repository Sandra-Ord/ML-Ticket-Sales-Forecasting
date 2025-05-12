using Microsoft.ML;
using Microsoft.ML.Data;
using ML.Domain;
using System.Data.SqlClient;

namespace ML.Data;

/// <summary>
/// Loads training data from a relational database using ML.NET's DatabaseLoader.
/// 
/// Src: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/load-data-ml-net
/// </summary>
public class DbTrainingDataLoader : ITrainingDataLoader
{
    private readonly string _connectionString;
    private readonly string _sqlQuery;

    /// <summary>
    /// Initializes a new instance of the <see cref="DbTrainingDataLoader"/> class.
    /// </summary>
    /// <param name="connectionString">Database connection string.</param>
    /// <param name="sqlQuery">SQL query used to fetch training data.</param>
    public DbTrainingDataLoader(string connectionString, string sqlQuery)
    {
        _connectionString = connectionString;
        _sqlQuery = sqlQuery;
    }

    /// <summary>
    /// Loads data from the database using the specified SQL query.
    /// </summary>
    /// <param name="mlContext">MLContext used to create the IDataView.</param>
    /// <returns>IDataView containing the training data from the database.</returns>
    public IDataView LoadData(MLContext mlContext)
    {
        Console.WriteLine("[DbLoader] Connecting to database...");

        var loader = mlContext.Data.CreateDatabaseLoader<CinemaAdmissionData>();

        var dbSource = new DatabaseSource(
            SqlClientFactory.Instance,
            _connectionString,
            _sqlQuery
        );

        return loader.Load(dbSource);
    }
}
