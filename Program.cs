using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Logit
{
    internal class LogisticProgram
    {
        private static readonly ILogger _logger = new ConsoleLogger();

        private static void Main(string[] args)
        {
            /*_logger.Info("Logistic regression using C# demo \n");

            _logger.Info("Raw data looks like: \n");
            _logger.Info("Female  66  mgmt  52100.00  low");
            _logger.Info("Male    35  tech  86100.00  medium");
            _logger.Info(" . . . \n");
            _logger.Info("Encoded and normed data looks like: \n");
            _logger.Info("1 - 0.66  1 0 0  0.5210  1 0 0"); //mgmt    low 
            _logger.Info("0 - 0.35  0 0 1  0.8610  0 1 0"); //tech    medium
            _logger.Info(" . . . \n");*/



            using var reader = File.OpenText("Data/employees_train.txt");
            var data = DataParser.Parse(reader);

            var (predictor, trainingPhases) = Predictor.TrainFrom(data);

            foreach (var tp in trainingPhases)
                _logger.Info($"iter = {tp.Epoch,5}  error = {tp.Error:F4} acc = {tp.Accuracy:F4}");

            _logger.Info($"Trained weights and bias: {string.Join(" ", predictor.Weights.Select(elem => elem.ToString("F4")))}");


            _logger.Info("Accuracy of model on training data: " + predictor.Accuracy.ToString("F4"));

            string xRaw = "36 tech $52,000.00 medium";
            _logger.Info("\nPredicting Sex for: ");
            _logger.Info(xRaw);
            double[] x = { 0.36, 0, 0, 1, 0.5200, 0, 1, 0 };
            double p = predictor.GetOutput(new PredictionInput(0.36, JobType.tech, 0.5200, Satisfaction.medium));
            _logger.Info($"Computed p-value = {p:F4}");
            _logger.Info($"Predicted Sex = {(p < 0.5 ? "Male" : "Female")}");

            Console.ReadLine();
        }
    }

    class Predictor
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

        //TODO change to IPredictionResult
        public double GetOutput(IPredictionInput input) => GetOutput(input.Encode(), _wts);

        public static (Predictor Predictor, IEnumerable<(int Epoch, double Error, double Accuracy)> TrainingPhases)
            TrainFrom(IReadOnlyList<(IPredictionInput Input, IPredictionResult Result)> data, double lr = 0.01, int maxEpoch = 100, IRandom rand = null)
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

            return (new Predictor(wts, acc, err), trainingPhases);
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


            for (int epoch = 0; epoch < maxEpoch; ++epoch)
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
                    int tmp = vec[ri];
                    vec[ri] = vec[i];
                    vec[i] = tmp;
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


    interface ILogger
    {
        void Info(string message);
    }
    class ConsoleLogger : ILogger
    {
        public void Info(string message) => Console.WriteLine(message);
    }
}