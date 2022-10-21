namespace Logit;

//TODO add anti-overfitting measures: https://en.wikipedia.org/wiki/Overfitting

class PredictorFactory
{
    public static TrainingSet<TInput, TResult> TrainFrom<TInput, TResult>(IReadOnlyList<(TInput Input, TResult Result)> data,
        Func<TInput, TInput> scallingFunction, PredictorConfiguration? configuration = null)
        where TInput : IPredictionInput
        where TResult : IBinaryResult
    {
        return Predictor<TInput, TResult>.TrainFrom(data, scallingFunction, configuration);
    }

    public static IReadOnlyCollection<TrainingSet<TInput, SimpleResult>> TrainMultipleClassFrom<TInput, TMultiClassResult>(
        IReadOnlyList<(TInput Input, TMultiClassResult Result)> data,
        Func<TInput, TInput> scallingFunction, PredictorConfiguration? configuration = null)
        where TInput : IPredictionInput
        where TMultiClassResult : IMultiClassResult<TMultiClassResult>
    {
        return Predictor<TInput, SimpleResult>.TrainMultipleClassFrom(data, scallingFunction, configuration);
    }
}

class Predictor<TInput, TResult>
    where TInput : IPredictionInput
    where TResult : IBinaryResult
{
    private readonly double[] _wts;
    private readonly Func<TInput, TInput> _scallingFunction;
    public IReadOnlyList<double> Weights => _wts;
    public double Accuracy { get; }
    public double Error { get; }

    private Predictor(double[] wts, Func<TInput, TInput> scallingFunction, double accuracy, double error)
    {
        _wts = wts;
        _scallingFunction = scallingFunction;
        Accuracy = accuracy;
        Error = error;
    }

    public double GetOutput(TInput input)
    {
        var encodedInput = _scallingFunction(input).Encode();
        return GetOutput(encodedInput, _wts);//pValue
    }

    public static TrainingSet<TInput, TResult> TrainFrom(IReadOnlyList<(TInput Input, TResult Result)> data,
        Func<TInput, TInput> scallingFunction, PredictorConfiguration? configuration = null)
    {
        var trainX = new double[data.Count][];
        var trainY = new int[data.Count];
        for (int i = 0; i < trainX.Length; i++)
        {
            var (input, result) = data[i];
            trainX[i] = scallingFunction(input).Encode();
            trainY[i] = result.Encode();
        }
        var wts = Train(trainX, trainY, out var trainingPhases, configuration);
        var err = GetError(trainX, trainY, wts);
        var acc = GetAccuracy(trainX, trainY, wts);

        return new(new Predictor<TInput, TResult>(wts, scallingFunction, acc, err), trainingPhases);
    }

    public static IReadOnlyCollection<TrainingSet<TInput, TResult>> TrainMultipleClassFrom<TMultiClassResult>(
        IReadOnlyList<(TInput Input, TMultiClassResult Result)> data,
        Func<TInput, TInput> scallingFunction, PredictorConfiguration? configuration = null)
        where TMultiClassResult : IMultiClassResult<TMultiClassResult>
    {
        var classesCount = data[0].Result.ClassCount;
        var list = new List<TrainingSet<TInput, TResult>>(classesCount);

        for (int classNumber = 0; classNumber < classesCount; classNumber++)
        {
            var trainX = new double[data.Count][];
            var trainY = new int[data.Count];
            for (int i = 0; i < trainX.Length; i++)
            {
                var (input, result) = data[i];
                trainX[i] = scallingFunction(input).Encode();
                trainY[i] = result.Separate(result, classNumber).Encode();
            }
            var wts = Train(trainX, trainY, out var trainingPhases, configuration);
            var err = GetError(trainX, trainY, wts);
            var acc = GetAccuracy(trainX, trainY, wts);

            var trainedSet = new TrainingSet<TInput, TResult>(new Predictor<TInput, TResult>(wts, scallingFunction, acc, err), trainingPhases);
            list.Add(trainedSet);
        }

        return list;
    }

    private static double[] Train(double[][] trainX, int[] trainY, out IEnumerable<TrainingPhase> trainingPhases, PredictorConfiguration? configuration = null)
    {
        configuration ??= new();

        var phases = new List<TrainingPhase>();

        int N = trainX.Length;  // number train items
        int n = trainX[0].Length;  // number predictors

        double lr = configuration.Lr;
        int maxEpoch = configuration.MaxEpoch;

        double[] wts = GenerateWeightsAndBias(n, -0.01, 0.01, configuration.Rand);

        int[] indices = Enumerable.Range(0, N).ToArray();


        for (int epoch = 0; epoch <= maxEpoch; ++epoch)
        {
            Shuffle(indices, configuration.Rand);

            foreach (int i in indices)
            {
                var x = trainX[i];  // predictors
                var y = trainY[i];  // target, 0 or 1
                var p = GetOutput(x, wts);

                for (int j = 0; j < n; ++j)  // each weight
                    wts[j] += lr * x[j] * (y - p) * p * (1 - p);
                wts[n] += lr * (y - p) * p * (1 - p);
            }

            if (epoch % (maxEpoch / 10) == 0)
            {
                double err = GetError(trainX, trainY, wts);
                double acc = GetAccuracy(trainX, trainY, wts);
                phases.Add(new(epoch, err, acc));
            }
        }

        trainingPhases = phases;

        return wts;  // trained weights and bias

        static void Shuffle(int[] vec, IRandom rnd)
        {
            int n = vec.Length;
            for (int i = 0; i < n; ++i)
            {
                int ri = rnd.Next(i, n);
                (vec[i], vec[ri]) = (vec[ri], vec[i]);
            }
        }
    }

    private static double GetAccuracy(double[][] dataX, int[] dataY, double[] wts)
    {
        int numCorrect = 0; int numWrong = 0;
        int N = dataX.Length;
        for (int i = 0; i < N; ++i)
        {
            var x = dataX[i];
            var y = dataY[i];  // actual, 0 or 1
            var p = GetOutput(x, wts);

            if (y == 0 && p < 0.5)
                ++numCorrect;
            else if (y == 1 && p >= 0.5)
                ++numCorrect;
            else
                ++numWrong;
        }
        return (1.0 * numCorrect) / (numCorrect + numWrong);
    }

    private static double GetError(double[][] dataX, int[] dataY, double[] wts)
    {
        double sum = 0.0;
        int N = dataX.Length;
        for (int i = 0; i < N; ++i)
        {
            var x = dataX[i];
            var y = dataY[i];  // target, 0 or 1
            var p = GetOutput(x, wts);
            sum += (p - y) * (p - y); // E = (o-t)^2 form
        }
        return sum / N;
    }

    private static double GetOutput(double[] x, double[] wts)
    {
        static double LogSig(double x) =>
            x switch
            {
                < -20.0 => 0.0,
                > 20.0 => 1.0,
                _ => 1.0 / (1.0 + Math.Exp(-x))
            };

        // bias is last cell of w
        double z = 0.0;
        for (int i = 0; i < x.Length; ++i)
            z += x[i] * wts[i];
        z += wts[^1];
        return LogSig(z);
    }

    private static double[] GenerateWeightsAndBias(int n, double lo, double hi, IRandom rand)
        => Enumerable.Repeat(0, n + 1).Select(_ => (hi - lo) * rand.NextDouble() + lo).ToArray();
}

readonly record struct TrainingPhase(int Epoch, double Error, double Accuracy);

record TrainingSet<TInput, TResult>(Predictor<TInput, TResult> Predictor, IEnumerable<TrainingPhase> TrainingPhases)
    where TInput : IPredictionInput
    where TResult : IBinaryResult;

class PredictorConfiguration
{
    public double Lr { get; }
    public int MaxEpoch { get; }
    public IRandom Rand { get; }

    public PredictorConfiguration(double lr = 0.01, int maxEpoch = 100, IRandom? rand = null)
    {
        Lr = lr;
        MaxEpoch = maxEpoch;
        Rand = rand ?? new SystemRandom(0);
    }
}