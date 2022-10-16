using Logit;


ILogger _logger = new ConsoleLogger();

using var reader = File.OpenText("Data/employees_train.txt");
var trainingData = new PersonDataParser().Parse(reader, out var scallingFunction);

var (predictor, trainingPhases) = Predictor<PersonInput, PersonResult>.TrainFrom(trainingData, maxEpoch:100);

foreach (var (Epoch, Error, Accuracy) in trainingPhases)
    _logger.Info($"iter = {Epoch,5}  error = {Error:F4} acc = {Accuracy:F4}");

_logger.Info($"Trained weights and bias: {string.Join(" ", predictor.Weights.Select(elem => elem.ToString("F4")))}");


_logger.Info("Accuracy of model on training data: " + predictor.Accuracy.ToString("F4"));


var input = new PersonInput(36, JobType.tech, 52000, Satisfaction.medium);
_logger.Info("\nPredicting Sex for: ");
_logger.Info(input.ToString());

double p = predictor.GetOutput(scallingFunction(input));
_logger.Info($"Computed p-value = {p:F4}");
_logger.Info($"Predicted Sex = {(p < 0.5 ? "Male" : "Female")}");

Console.ReadLine();
