using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace ML.Model;

public static class FastTreeDefinition
{
    public static IEstimator<ITransformer> CreateTrainer(MLContext context)
    {
        //var options = new FastTreeTrainer.Options
        //{
        //    NumberOfTrees = 1000,
        //    NumberOfLeaves = 200,
        //    MinimumExampleCountPerLeaf = 50,
        //    LearningRate = 0.01f
        //};

        return context.Regression.Trainers.FastTree(numberOfTrees: 1000, numberOfLeaves: 200, minimumExampleCountPerLeaf: 50, learningRate: 0.01f);
    }

    public static IEstimator<ITransformer> CreatePipeline(MLContext context)
    {
        return PipelineBuilder.BuildPipeLine(context, normalized: false, logUsed: false)
                              .Append(CreateTrainer(context));
    }
}
