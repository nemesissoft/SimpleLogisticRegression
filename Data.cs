using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Logit;

static partial class DataParser
{
    private static readonly Regex _linePattern = LinePattern();
    [GeneratedRegex(@"[\w\.]+  |  ""[\w\.\s]*""", RegexOptions.Compiled | RegexOptions.IgnorePatternWhitespace)]
    private static partial Regex LinePattern();

    public static IReadOnlyList<(IPredictionInput Input, IPredictionResult Result)> Parse(StreamReader reader)
    {
        var trans = Nemesis.TextParsers.TextTransformer.Default;
        T Parse<T>(string text) => trans.GetTransformer<T>().Parse(text);
        static double RoundUpToNearestPowerOf10(double value) => Math.Pow(10, Math.Ceiling(Math.Log10(value)));

        var lines = new List<(bool IsFemale, double Age, JobType Job, double Income, Satisfaction Satisfaction)>();

        while (reader.ReadLine() is { } line)
        {
            if (line.StartsWith('#')) continue;

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

        //TODO add scaling method

        maxAge = RoundUpToNearestPowerOf10(maxAge);
        maxIncome = RoundUpToNearestPowerOf10(maxIncome);

        var result = new List<(IPredictionInput Input, IPredictionResult Result)>(lines.Count);

        foreach (var (isFemale, age, job, income, satisfaction) in lines)
        {
            result.Add(new(
                new PredictionInput(age / maxAge, job, income / maxIncome, satisfaction),
                new PredictionResult(!isFemale) //TODO rewert
                ));
        }

        return result;
    }    
}

interface IPredictionInput
{
    double[] Encode();
}

interface IPredictionResult
{
    double Encode();
}


readonly record struct PredictionInput(double Age, JobType Job, double Income, Satisfaction Satisfaction) : IPredictionInput
{
    public double[] Encode()
    {
        //Female  66  mgmt  52100.00  low
        //Male    35  tech  86100.00  medium
        //1 - 0.66  1 0 0  0.5210  1 0 0
        //0 - 0.35  0 0 1  0.8610  0 1 0

        var result = new double[8];

        void Mark(int enumInt, int startingIndex) => result[startingIndex + enumInt] = 1.0;

        result[0] = Age;

        Mark((int)Job, 1);

        result[4] = Income;

        Mark((int)Satisfaction, 5);

        return result;
    }
}

readonly record struct PredictionResult(bool IsFemale) : IPredictionResult
{
    public double Encode() => IsFemale ? 1.0 : 0.0;
}

enum JobType { mgmt, sale, tech }
enum Satisfaction { low, medium, high }

