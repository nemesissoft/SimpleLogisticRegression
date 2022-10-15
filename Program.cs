using System;
using System.Collections.Generic;
using System.Linq;

namespace Logit
{
    internal class LogisticProgram
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("Logistic regression using C# demo \n");

            Console.WriteLine("Raw data looks like: \n");
            Console.WriteLine("Female  66  mgmt  52100.00  low");
            Console.WriteLine("Male    35  tech  86100.00  medium");
            Console.WriteLine(" . . . \n");
            Console.WriteLine("Encoded and normed data looks like: \n");
            Console.WriteLine("1 - 0.66  1 0 0  0.5210  1 0 0"); //mgmt    low 
            Console.WriteLine("0 - 0.35  0 0 1  0.8610  0 1 0"); //tech    medium
            Console.WriteLine(" . . . \n");

            // load data. 23 Male (0), 17 Female (1)
            double[][] trainX = new double[40][];
            trainX[0] = new[] { 0.66, 1, 0, 0, 0.5210, 1, 0, 0 };
            trainX[1] = new[] { 0.35, 0, 0, 1, 0.8610, 0, 1, 0 };
            trainX[2] = new[] { 0.24, 0, 0, 1, 0.4410, 0, 0, 1 };
            trainX[3] = new[] { 0.43, 0, 1, 0, 0.5170, 0, 1, 0 };
            trainX[4] = new[] { 0.37, 1, 0, 0, 0.8860, 0, 1, 0 };
            trainX[5] = new[] { 0.30, 0, 1, 0, 0.8790, 1, 0, 0 };
            trainX[6] = new[] { 0.40, 1, 0, 0, 0.2020, 0, 1, 0 };
            trainX[7] = new[] { 0.58, 0, 0, 1, 0.2650, 1, 0, 0 };
            trainX[8] = new[] { 0.27, 1, 0, 0, 0.8480, 1, 0, 0 };
            trainX[9] = new[] { 0.33, 0, 1, 0, 0.5600, 0, 1, 0 };
            trainX[10] = new[] { 0.59, 0, 0, 1, 0.2330, 0, 0, 1 };
            trainX[11] = new[] { 0.52, 0, 1, 0, 0.8700, 0, 0, 1 };
            trainX[12] = new[] { 0.41, 1, 0, 0, 0.5170, 0, 1, 0 };
            trainX[13] = new[] { 0.22, 0, 1, 0, 0.3500, 0, 0, 1 };
            trainX[14] = new[] { 0.61, 0, 1, 0, 0.2980, 1, 0, 0 };
            trainX[15] = new[] { 0.46, 1, 0, 0, 0.6780, 0, 1, 0 };
            trainX[16] = new[] { 0.59, 1, 0, 0, 0.8430, 1, 0, 0 };
            trainX[17] = new[] { 0.28, 0, 0, 1, 0.7730, 0, 0, 1 };
            trainX[18] = new[] { 0.46, 0, 1, 0, 0.8930, 0, 1, 0 };
            trainX[19] = new[] { 0.48, 0, 0, 1, 0.2920, 0, 1, 0 };
            trainX[20] = new[] { 0.28, 1, 0, 0, 0.6690, 0, 1, 0 };
            trainX[21] = new[] { 0.23, 0, 1, 0, 0.8970, 0, 0, 1 };
            trainX[22] = new[] { 0.60, 1, 0, 0, 0.6270, 0, 0, 1 };
            trainX[23] = new[] { 0.29, 0, 1, 0, 0.7760, 1, 0, 0 };
            trainX[24] = new[] { 0.24, 0, 0, 1, 0.8750, 0, 0, 1 };
            trainX[25] = new[] { 0.51, 1, 0, 0, 0.4090, 0, 1, 0 };
            trainX[26] = new[] { 0.22, 0, 1, 0, 0.8910, 1, 0, 0 };
            trainX[27] = new[] { 0.19, 0, 0, 1, 0.5380, 1, 0, 0 };
            trainX[28] = new[] { 0.25, 0, 1, 0, 0.9000, 0, 0, 1 };
            trainX[29] = new[] { 0.44, 0, 0, 1, 0.8980, 0, 1, 0 };
            trainX[30] = new[] { 0.35, 1, 0, 0, 0.5380, 0, 1, 0 };
            trainX[31] = new[] { 0.29, 0, 1, 0, 0.7610, 1, 0, 0 };
            trainX[32] = new[] { 0.25, 1, 0, 0, 0.3450, 0, 1, 0 };
            trainX[33] = new[] { 0.66, 1, 0, 0, 0.2210, 1, 0, 0 };
            trainX[34] = new[] { 0.43, 0, 0, 1, 0.7450, 0, 1, 0 };
            trainX[35] = new[] { 0.42, 0, 1, 0, 0.8520, 0, 1, 0 };
            trainX[36] = new[] { 0.44, 1, 0, 0, 0.6580, 0, 1, 0 };
            trainX[37] = new[] { 0.42, 0, 1, 0, 0.6970, 0, 1, 0 };
            trainX[38] = new[] { 0.56, 0, 0, 1, 0.3680, 0, 0, 1 };
            trainX[39] = new[] { 0.38, 1, 0, 0, 0.2600, 1, 0, 0 };

            double[] trainY =
            {
                1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1,
                1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0
            };

            int maxEpoch = 100;
            double lr = 0.01;
            Console.WriteLine("SGD training with lr = 0.01");
            double[] wts = Train(trainX, trainY, lr, maxEpoch, 0);
            Console.WriteLine("Training complete \n");

            Console.WriteLine("Trained weights and bias: ");
            ShowVector(wts);

            double accTrain = Accuracy(trainX, trainY, wts);
            Console.Write("Accuracy of model on training data: ");
            Console.WriteLine(accTrain.ToString("F4"));

            string xRaw = "36 tech $52,000.00 medium";
            Console.WriteLine("\nPredicting Sex for: ");
            Console.WriteLine(xRaw);
            double[] x = { 0.36, 0, 0, 1, 0.5200, 0, 1, 0 };
            double p = ComputeOutput(x, wts);
            Console.WriteLine($"Computed p-value = {p:F4}");
            Console.WriteLine($"Predicted Sex = {(p < 0.5 ? "Male" : "Female")}");

            Console.ReadLine();
        }

        private static double ComputeOutput(double[] x, double[] wts)
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

        private static double[] Train(double[][] trainX, double[] trainY, double lr, int maxEpoch, int seed = 0)
        {
            int N = trainX.Length;  // number train items
            int n = trainX[0].Length;  // number predictors

            var rand = new SystemRandom(seed);
            double[] wts = GenerateWeightsAndBias(n, -0.01, 0.01, rand);

            int[] indices = Enumerable.Range(0, N).ToArray();


            for (int epoch = 0; epoch < maxEpoch; ++epoch)
            {
                Shuffle(indices, rand);

                foreach (int i in indices)
                {
                    double[] x = trainX[i];  // predictors
                    double y = trainY[i];  // target, 0 or 1
                    double p = ComputeOutput(x, wts);

                    for (int j = 0; j < n; ++j)  // each weight
                        wts[j] += lr * x[j] * (y - p) * p * (1 - p);
                    wts[n] += lr * (y - p) * p * (1 - p);
                }

                if (epoch % (maxEpoch / 10) == 0)
                {
                    double err = Error(trainX, trainY, wts);
                    double acc = Accuracy(trainX, trainY, wts);
                    Console.WriteLine($"iter = {epoch,5}  error = {err:F4} acc = {acc:F4}");
                }
            }

            return wts;  // trained weights and bias
        }

        private static double[] GenerateWeightsAndBias(int n, double lo, double hi, IRandom rand)
            => Enumerable.Repeat(0, n + 1).Select(_ => (hi - lo) * rand.NextDouble() + lo).ToArray();

        private static void Shuffle(int[] vec, IRandom rnd)
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

        private static double Accuracy(double[][] dataX, double[] dataY, double[] wts)
        {
            int numCorrect = 0; int numWrong = 0;
            int N = dataX.Length;
            for (int i = 0; i < N; ++i)
            {
                double[] x = dataX[i];
                double y = dataY[i];  // actual, 0 or 1
                double p = ComputeOutput(x, wts);

                if (y == 0 && p < 0.5)
                    ++numCorrect;
                else if (y == 1 && p >= 0.5)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (1.0 * numCorrect) / (numCorrect + numWrong);
        }

        private static double Error(double[][] dataX, double[] dataY, double[] wts)
        {
            double sum = 0.0;
            int N = dataX.Length;
            for (int i = 0; i < N; ++i)
            {
                double[] x = dataX[i];
                double y = dataY[i];  // target, 0 or 1
                double p = ComputeOutput(x, wts);
                sum += (p - y) * (p - y); // E = (o-t)^2 form
            }
            return sum / N;
        }

        private static void ShowVector(IEnumerable<double> vector)
            => Console.WriteLine(string.Join(" ", vector.Select(elem => elem.ToString("F4"))));
    }

    

    
}