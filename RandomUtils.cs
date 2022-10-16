using System.Numerics;

namespace Logit;

interface IRandom<TNumber>
    where TNumber : IBinaryFloatingPointIeee754<TNumber>
{
    double NextDouble();

    TNumber NextFloatingPoint() => TNumber.CreateSaturating(NextDouble());

    int Next(int minValue, int maxValue);
}

class SystemRandom<TNumber> : IRandom<TNumber>
    where TNumber : IBinaryFloatingPointIeee754<TNumber>
{
    private readonly Random _rand;

    public SystemRandom(int seed = 0) => _rand = new Random(seed);

    public double NextDouble() => _rand.NextDouble();

    public int Next(int minValue, int maxValue) => _rand.Next(minValue, maxValue);
}
