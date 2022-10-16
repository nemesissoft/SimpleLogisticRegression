namespace Logit;

interface IRandom
{
    double NextDouble();
    int Next(int minValue, int maxValue);
}

class SystemRandom : IRandom
{
    private readonly Random _rand;

    public SystemRandom(int seed = 0) => _rand = new Random(seed);

    public double NextDouble() => _rand.NextDouble();

    public int Next(int minValue, int maxValue) => _rand.Next(minValue, maxValue);
}
