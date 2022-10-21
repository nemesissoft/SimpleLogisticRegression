namespace Logit;

interface IDataParser<TInput, TResult>
    where TInput : IPredictionInput
{
    IReadOnlyList<(TInput Input, TResult TResult)> Parse(StreamReader reader, out Func<TInput, TInput> scallingFunction);
}

interface IPredictionInput
{
    double[] Encode();
}

interface IBinaryResult
{
    int Encode();
}

interface IBinaryOtputDecoder<T> where T : IBinaryResult
{
    T Decode(double probability);
}