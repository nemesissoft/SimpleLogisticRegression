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
interface IBinaryResultDecoder<T> where T : IBinaryResult
{
    T Decode(double probability);
}



//OvR
interface IMultiClassResult<TSelf> where TSelf : IMultiClassResult<TSelf>
{
    int ClassCount { get; }
    SimpleResult Separate(TSelf multiClass, int classNumber);
}
readonly record struct SimpleResult(bool Value) : IBinaryResult
{
    public int Encode() => Value ? 1 : 0;
}
interface IMultiClassResultDecoder<T> where T : IMultiClassResult<T>
{
    T Decode(IReadOnlyList<double> probabilities);
}