using System.Text.RegularExpressions;

namespace Logit;

class PersonSatisfactionDataParser : IDataParser<PersonSatisfactionInput, PersonSatisfactionResult>
{
    private static readonly Regex _linePattern = new(@"[\w\.]+  |  ""[\w\.\s]*""", RegexOptions.Compiled | RegexOptions.IgnorePatternWhitespace);

    public IReadOnlyList<(PersonSatisfactionInput Input, PersonSatisfactionResult TResult)> Parse(StreamReader reader, out Func<PersonSatisfactionInput, PersonSatisfactionInput> scallingFunction)
    {
        var trans = Nemesis.TextParsers.TextTransformer.Default;
        T Parse<T>(string text) => trans.GetTransformer<T>().Parse(text);
        static double RoundUpToNearestPowerOf10(double value) => Math.Pow(10, Math.Ceiling(Math.Log10(value)));

        var lines = new List<(bool IsContractor, double Age, JobType Job, double Income, Satisfaction Satisfaction)>();

        while (reader.ReadLine() is { } line)
        {
            if (line.StartsWith('#') || string.IsNullOrWhiteSpace(line)) continue;

            var matches = _linePattern.Matches(line);

            if (matches.Count != 5)
                throw new($"Expected 5 captures in line: \n{line}");


            lines.Add((
                Parse<bool>(matches[0].Value),
                Parse<double>(matches[1].Value),
                Parse<JobType>(matches[2].Value),
                Parse<double>(matches[3].Value),
                Parse<Satisfaction>(matches[4].Value)
                ));
        }
        var maxAge = lines.Max(t => t.Age);
        var maxIncome = lines.Max(t => t.Income);

        scallingFunction = input => input with
        {
            Age = input.Age / maxAge,
            Income = input.Income / maxIncome
        };

        maxAge = RoundUpToNearestPowerOf10(maxAge);
        maxIncome = RoundUpToNearestPowerOf10(maxIncome);

        var result = new List<(PersonSatisfactionInput Input, PersonSatisfactionResult Result)>(lines.Count);

        foreach (var (isContractor, age, job, income, satisfaction) in lines)
        {
            result.Add(new(
                new PersonSatisfactionInput(isContractor, age, job, income),
                new PersonSatisfactionResult(satisfaction)
                ));
        }

        return result;
    }
}

readonly record struct PersonSatisfactionInput(bool IsContractor, double Age, JobType Job, double Income) : IPredictionInput
{
    public double[] Encode()
    {
        var result = new double[6];

        void Mark(int enumInt, int startingIndex) => result[startingIndex + enumInt] = 1.0;

        result[0] = IsContractor ? 1.0 : 0.0;

        result[1] = Age;

        Mark((int)Job, 2);

        result[5] = Income;

        return result;
    }
}

readonly record struct PersonSatisfactionResult(Satisfaction Satisfaction) : IMultiClassResult<PersonSatisfactionResult>
{
    public int ClassCount => 3;

    public SimpleResult Separate(PersonSatisfactionResult multiClass, int classNumber) => classNumber switch
    {
        0 => new(multiClass.Satisfaction == Satisfaction.low),
        1 => new(multiClass.Satisfaction == Satisfaction.medium),
        2 => new(multiClass.Satisfaction == Satisfaction.high),
        _ => throw new NotSupportedException("Only classes 0 to 2 are supported")
    };
}

class PersonSatisfactionDecoder : IMultiClassResultDecoder<PersonSatisfactionResult>
{
    private PersonSatisfactionDecoder() { }

    public static IMultiClassResultDecoder<PersonSatisfactionResult> Instance { get; } = new PersonSatisfactionDecoder();

    public PersonSatisfactionResult Decode(IReadOnlyList<double> probabilities)
    {
        var max = probabilities.Select((prob, index) => (prob, index)).MaxBy(t => t.prob);

        return new((Satisfaction)max.index);
    }
}