namespace Logit;

class Predictor<TInput, TResult>
    where TInput : IPredictionInput 
    where TResult : IPredictionResult
{
    private readonly double[] _wts;
    public IReadOnlyList<double> Weights => _wts;
    public double Accuracy { get; }
    public double Error { get; }

    private Predictor(double[] wts, double accuracy, double error)
    {
        _wts = wts;
        Accuracy = accuracy;
        Error = error;
    }

    //TODO change to TResult
    public double GetOutput(TInput input) => GetOutput(input.Encode(), _wts);

    public static (Predictor<TInput, TResult> Predictor, IEnumerable<(int Epoch, double Error, double Accuracy)> TrainingPhases)
        TrainFrom(IReadOnlyList<(TInput Input, TResult Result)> data, double lr = 0.01, int maxEpoch = 100, IRandom rand = null)        
    {
        double[][] trainX = new double[data.Count][];
        double[] trainY = new double[data.Count];
        for (int i = 0; i < trainX.Length; i++)
        {
            var d = data[i];
            trainX[i] = d.Input.Encode();
            trainY[i] = d.Result.Encode();
        }
        double[] wts = Train(trainX, trainY, out var trainingPhases, lr, maxEpoch, rand);
        double err = GetError(trainX, trainY, wts);
        double acc = GetAccuracy(trainX, trainY, wts);

        return (new Predictor<TInput, TResult>(wts, acc, err), trainingPhases);
    }

    private static double[] Train(double[][] trainX, double[] trainY, out IEnumerable<(int Epoch, double Error, double Accuracy)> trainingPhases,
        double lr = 0.01, int maxEpoch = 100, IRandom rand = null)
    {
        var phases = new List<(int Epoch, double Error, double Accuracy)>();

        int N = trainX.Length;  // number train items
        int n = trainX[0].Length;  // number predictors

        rand ??= new SystemRandom(0);
        double[] wts = GenerateWeightsAndBias(n, -0.01, 0.01, rand);

        int[] indices = Enumerable.Range(0, N).ToArray();


        for (int epoch = 0; epoch <= maxEpoch; ++epoch)
        {
            Shuffle(indices, rand);

            foreach (int i in indices)
            {
                double[] x = trainX[i];  // predictors
                double y = trainY[i];  // target, 0..1
                double p = GetOutput(x, wts);

                for (int j = 0; j < n; ++j)  // each weight
                    wts[j] += lr * x[j] * (y - p) * p * (1 - p);
                wts[n] += lr * (y - p) * p * (1 - p);
            }

            if (epoch % (maxEpoch / 10) == 0)
            {
                double err = GetError(trainX, trainY, wts);
                double acc = GetAccuracy(trainX, trainY, wts);
                phases.Add((epoch, err, acc));
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

    private static double GetAccuracy(double[][] dataX, double[] dataY, double[] wts)
    {
        int numCorrect = 0; int numWrong = 0;
        int N = dataX.Length;
        for (int i = 0; i < N; ++i)
        {
            double[] x = dataX[i];
            double y = dataY[i];  // actual, 0 or 1
            double p = GetOutput(x, wts);

            if (y == 0 && p < 0.5)
                ++numCorrect;
            else if (y == 1 && p >= 0.5)
                ++numCorrect;
            else
                ++numWrong;
        }
        return (1.0 * numCorrect) / (numCorrect + numWrong);
    }

    private static double GetError(double[][] dataX, double[] dataY, double[] wts)
    {
        double sum = 0.0;
        int N = dataX.Length;
        for (int i = 0; i < N; ++i)
        {
            double[] x = dataX[i];
            double y = dataY[i];  // target, 0 or 1
            double p = GetOutput(x, wts);
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