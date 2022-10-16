namespace Logit;

interface IDataParser<TInput, TResult> where TInput : IPredictionInput where TResult : IPredictionResult
{
    IReadOnlyList<(TInput Input, TResult TResult)> Parse(StreamReader reader, out Func<TInput, TInput> scallingFunction);
}

interface IPredictionInput
{
    double[] Encode();
}

interface IPredictionResult
{
    double Encode();
}