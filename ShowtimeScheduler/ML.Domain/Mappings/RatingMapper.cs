using Microsoft.ML.Transforms;
using ML.Domain.Features;

namespace ML.Domain.Mappings;

[CustomMappingFactoryAttribute(nameof(RatingMapper))]
public class RatingMapper : CustomMappingFactory<CinemaAdmissionData, EventOrdinalRating>
{
    private static readonly Dictionary<string, float> _ratingMap = new()
        {
            { "Lubatud kõigile", 0f },
            { "Perefilm", 0f },
            { "Alla 6 a. mittesoovitatav", 6f },
            { "Alla 12 a. mittesoovitatav", 12f },
            { "Alla 12 a. keelatud", 12f },
            { "Alla 14 a. keelatud", 14f },
            { "Alla 16 a. keelatud", 16f },
            { "Alla 18 a. keelatud", 18f }
        };

    public override Action<CinemaAdmissionData, EventOrdinalRating> GetMapping()
    {
        return (input, output) =>
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (output is null) throw new ArgumentNullException(nameof(output));

            if (!_ratingMap.TryGetValue(input.EventRating, out var encoded))
                encoded = 0f; // fallback

            output.EventRatingEncoded = encoded;
        };
    }
}