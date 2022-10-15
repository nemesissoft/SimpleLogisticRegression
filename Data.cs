using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Logit;

static class DataParser
{
    private static readonly Regex _linePattern = new(@"[\w\.]+  |  ""[\w\.\s]*""", RegexOptions.Compiled | RegexOptions.IgnorePatternWhitespace);

    public static IReadOnlyCollection<(PredictionInput Input, PredictionResult Result)> Parse(StreamReader reader)
    {
        var trans = Nemesis.TextParsers.TextTransformer.Default;
        T Parse<T>(string text) => trans.GetTransformer<T>().Parse(text);
        static double RoundUpToNearestPowerOf10(double value) => Math.Pow(10, Math.Ceiling(Math.Log10(value)));

        var lines = new List<(bool IsFemale, double Age, JobType Job, double Income, Satisfaction Satisfaction)>();
        
        while (reader.ReadLine() is { } line)
        {
            if (line.StartsWith('#')) continue;

            var match = _linePattern.Match(line);
            if (!match.Success || match.Captures.Count != 5)
                throw new($"Expected 5 captures in line: \n{line}");
            var captures = match.Captures;

            lines.Add((
                Parse<bool>(captures[0].Value),
                Parse<double>(captures[1].Value),
                Parse<JobType>(captures[2].Value),
                Parse<double>(captures[3].Value),
                Parse<Satisfaction>(captures[4].Value)
                ));
        }
        var maxAge = lines.Max(t => t.Age);
        var maxIncome = lines.Max(t => t.Income);



        var result = new List<(PredictionInput Input, PredictionResult Result)>(lines.Count);

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

