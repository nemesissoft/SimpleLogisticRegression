namespace Logit;

interface IDataParser<TInput, TResult> 
    where TInput : IPredictionInput 
    where TResult : IPredictionResult<TResult>
{
    IReadOnlyList<(TInput Input, TResult TResult)> Parse(StreamReader reader, out Func<TInput, TInput> scallingFunction);
}

interface IPredictionInput
{
    double[] Encode();
}

interface IPredictionResult<TSelf> 
    where TSelf : IPredictionResult<TSelf>
{
    double Encode();

    static abstract TSelf Parse(double probability);
}