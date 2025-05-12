namespace App.DTO;

/// <summary>
/// Represents the result of a prediction, including the value, errors, and warnings.
/// </summary>
/// <typeparam name="T">Type of the prediction result value.</typeparam>
public class PredictionResult<T>
{
    /// <summary>
    /// Predicted result, if successful.
    /// </summary>
    public T? Value { get; set; }

    /// <summary>
    /// List of error messages if the prediction failed.
    /// </summary>
    public List<string> Errors { get; set; } = new();

    /// <summary>
    /// List of warning messages if the prediction succeeded with concerns.
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Indicates whether the prediction was successful (no errors).
    /// </summary>
    public bool IsSuccess => !Errors.Any();
}