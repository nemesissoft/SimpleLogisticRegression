using System;
using System.Numerics;

namespace Logit;

class Predictor<TInput, TResult, TNumber>
    where TInput : IPredictionInput<TNumber>
    where TResult : IPredictionResult<TResult, TNumber>
    where TNumber : IBinaryFloatingPointIeee754<TNumber>
{
    private readonly TNumber[] _wts;
    private readonly Func<TInput, TInput> _scallingFunction;
    public IReadOnlyList<TNumber> Weights => _wts;
    public TNumber Accuracy { get; }
    public TNumber Error { get; }

    private Predictor(TNumber[] wts, Func<TInput, TInput> scallingFunction, TNumber accuracy, TNumber error)
    {
        _wts = wts;
        _scallingFunction = scallingFunction;
        Accuracy = accuracy;
        Error = error;
    }

    public TResult GetOutput(TInput input, out TNumber pValue)
    {
        TNumber[] encodedInput = _scallingFunction(input).Encode();

        pValue = GetOutput(encodedInput, _wts);

        return TResult.Parse(pValue);
    }

    public static (Predictor<TInput, TResult, TNumber> Predictor, IEnumerable<TrainingPhase> TrainingPhases)
        TrainFrom(IReadOnlyList<(TInput Input, TResult Result)> data, Func<TInput, TInput> scallingFunction, PredictorConfiguration configuration)
    {
        var trainX = new TNumber[data.Count][];
        var trainY = new TNumber[data.Count];
        for (int i = 0; i < trainX.Length; i++)
        {
            var (input, result) = data[i];
            trainX[i] = scallingFunction(input).Encode();
            trainY[i] = result.Encode();
        }
        var wts = Train(trainX, trainY, out var trainingPhases, configuration);
        var err = GetError(trainX, trainY, wts);
        var acc = GetAccuracy(trainX, trainY, wts);

        return (new Predictor<TInput, TResult, TNumber>(wts, scallingFunction, acc, err), trainingPhases);
    }

    private static TNumber[] Train(TNumber[][] trainX, TNumber[] trainY, out IEnumerable<TrainingPhase> trainingPhases, PredictorConfiguration configuration)
    {
        var phases = new List<TrainingPhase>();

        int N = trainX.Length;  // number train items
        int n = trainX[0].Length;  // number predictors

        var lr = TNumber.CreateSaturating(configuration.Lr);
        int maxEpoch = configuration.MaxEpoch;

        var wts = GenerateWeightsAndBias(n, TNumber.CreateSaturating(-0.01), TNumber.CreateSaturating(0.01), configuration.Rand);

        int[] indices = Enumerable.Range(0, N).ToArray();


        for (int epoch = 0; epoch <= maxEpoch; ++epoch)
        {
            Shuffle(indices, configuration.Rand);

            foreach (int i in indices)
            {
                var x = trainX[i];  // predictors
                var y = trainY[i];  // target, 0..1
                var p = GetOutput(x, wts);

                for (int j = 0; j < n; ++j)  // each weight
                    wts[j] += lr * x[j] * (y - p) * p * (TNumber.One - p);
                wts[n] += lr * (y - p) * p * (TNumber.One - p);
            }

            if (epoch % (maxEpoch / 10) == 0)
            {
                var err = GetError(trainX, trainY, wts);
                var acc = GetAccuracy(trainX, trainY, wts);
                phases.Add(new(epoch, err, acc));
            }
        }

        trainingPhases = phases;

        return wts;  // trained weights and bias

        static void Shuffle(int[] vec, IRandom<TNumber> rnd)
        {
            int n = vec.Length;
            for (int i = 0; i < n; ++i)
            {
                int ri = rnd.Next(i, n);
                (vec[i], vec[ri]) = (vec[ri], vec[i]);
            }
        }
    }

    private static TNumber GetAccuracy(TNumber[][] dataX, TNumber[] dataY, TNumber[] wts)
    {
        int numCorrect = 0; int numWrong = 0;
        int N = dataX.Length;
        var half = TNumber.CreateSaturating(0.5);

        for (int i = 0; i < N; ++i)
        {
            var x = dataX[i];
            var y = dataY[i];  // actual, 0 or 1
            var p = GetOutput(x, wts);



            if (y == TNumber.Zero && p < half)
                ++numCorrect;
            else if (y == TNumber.One && p >= half)
                ++numCorrect;
            else
                ++numWrong;
        }
        return TNumber.CreateSaturating((1.0 * numCorrect) / (numCorrect + numWrong));
    }

    private static TNumber GetError(TNumber[][] dataX, TNumber[] dataY, TNumber[] wts)
    {
        TNumber sum = TNumber.Zero;
        int N = dataX.Length;
        for (int i = 0; i < N; ++i)
        {
            var x = dataX[i];
            var y = dataY[i];  // target, 0 or 1
            var p = GetOutput(x, wts);
            sum += (p - y) * (p - y); // E = (o-t)^2 form
        }
        return sum / TNumber.CreateSaturating(N);
    }

    private static TNumber GetOutput(TNumber[] x, TNumber[] wts)
    {
        static TNumber LogSig(TNumber x) =>
            x switch
            {
                < -20.0 => TNumber.Zero,
                > 20.0 => TNumber.One,
                _ => TNumber.One / (TNumber.One + TNumber.Exp(-x))
            };

        // bias is last cell of w
        TNumber z = TNumber.Zero;
        for (int i = 0; i < x.Length; ++i)
            z += x[i] * wts[i];
        z += wts[^1];
        return LogSig(z);
    }

    private static TNumber[] GenerateWeightsAndBias(int n, TNumber lo, TNumber hi, IRandom<TNumber> rand) =>
        Enumerable.Repeat(0, n + 1).Select(_ => (hi - lo) * rand.NextFloatingPoint() + lo).ToArray();

    public readonly record struct TrainingPhase(int Epoch, TNumber Error, TNumber Accuracy);

    public readonly struct PredictorConfiguration
    {
        public double Lr { get; }
        public int MaxEpoch { get; }
        public IRandom<TNumber> Rand { get; }

        public PredictorConfiguration(double lr = 0.01, int maxEpoch = 100, IRandom<TNumber> rand = null)
        {
            Lr = lr;
            MaxEpoch = maxEpoch;
            Rand = rand ?? new SystemRandom<TNumber>(0);
        }
    }
}



