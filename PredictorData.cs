using System.Numerics;

namespace Logit;

interface IDataParser<TInput, TResult, TNumber>
    where TInput : IPredictionInput<TNumber>
    where TResult : IPredictionResult<TResult, TNumber>
    where TNumber : IBinaryFloatingPointIeee754<TNumber>
{
    IReadOnlyList<(TInput Input, TResult TResult)> Parse(StreamReader reader, out Func<TInput, TInput> scallingFunction);
}


interface IPredictionInput<TNumber> 
    where TNumber : IBinaryFloatingPointIeee754<TNumber>
{
    TNumber[] Encode();
}


interface IPredictionResult<TSelf, TNumber>
    where TSelf : IPredictionResult<TSelf, TNumber>
    where TNumber : IBinaryFloatingPointIeee754<TNumber>
{
    TNumber Encode();

    static abstract TSelf Parse(TNumber probability);
}