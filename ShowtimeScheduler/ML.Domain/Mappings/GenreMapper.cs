using Microsoft.ML.Transforms;
using ML.Domain.Features;

namespace ML.Domain.Mappings;

[CustomMappingFactoryAttribute(nameof(GenreMapper))]
public class GenreMapper : CustomMappingFactory<CinemaAdmissionData, EventGenres>
{
    public override Action<CinemaAdmissionData, EventGenres> GetMapping() 
    {
        return (input, output) =>
        {
            output.EventGenresArray = input.EventGenres
            .Split(new[] { ", " }, StringSplitOptions.RemoveEmptyEntries)
            .Select(g => g.Trim())
            .ToArray();
        };
    }
}