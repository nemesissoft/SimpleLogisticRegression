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
            using var reader = File.OpenText("Data/employees_train.txt");
            var trainingData = DataParser.Parse(reader);

            var (predictor, trainingPhases) = Predictor.TrainFrom(trainingData);

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

   


    
}