using Microsoft.Extensions.Hosting;

namespace ML.Services;

/// <summary>
/// Background service that periodically triggers model (re)training.
/// </summary>
public class TrainingScheduler : BackgroundService
{
    private readonly TrainerService _trainer;

    /// <summary>
    /// Initializes a new instance of the <see cref="TrainingScheduler"/> class.
    /// </summary>
    /// <param name="trainer">Trainer service responsible for model retraining</param>
    public TrainingScheduler(TrainerService trainer)
    {
        _trainer = trainer;
    }

    /// <summary>
    /// Executes the background service loop which schedules and triggers model retraining.
    /// </summary>
    /// <param name="stoppingToken">A token to monitor for shutdown requests.</param>
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            var now = DateTime.Now;
            var nextRun = GetNextMondayAtHour(2);
            var waitTime = nextRun - now;
            Console.WriteLine($"[TrainingScheduler] Next retrain scheduled for {nextRun}");

            try
            {
                //await Task.Delay(waitTime, stoppingToken);

                await Task.Delay(TimeSpan.FromMinutes(7), stoppingToken);

                Console.WriteLine("[TrainingScheduler] Starting scheduled retraining...");
                _trainer.TrainAndSaveModel();

            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TrainingScheduler] Error: {ex.Message}");
                await Task.Delay(TimeSpan.FromMinutes(10), stoppingToken); // fallback wait
            }
        }
    }

    /// <summary>
    /// Calculates the next Monday at a specified hour.
    /// </summary>
    /// <param name="hour">The hour (in 24-hour format) for the training to run - preferably during a time with a small to none demand.</param>
    /// <returns>Date and time of the next scheduled training time.</returns>
    private DateTime GetNextMondayAtHour(int hour)
    {
        var now = DateTime.Now;
        int daysUntilMonday = ((int)DayOfWeek.Monday - (int)now.DayOfWeek + 7) % 7;
        if (daysUntilMonday == 0 && now.Hour >= hour)
            daysUntilMonday = 7;

        var nextMonday = now.Date.AddDays(daysUntilMonday).AddHours(hour);
        return nextMonday;
    }
}