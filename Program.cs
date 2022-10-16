using Logit;

ILogger logger = new ConsoleLogger();

logger.Info("EMPLOYMENT");
EmploymentPrediction(logger);


logger.Info("\n\n\n\n\n\n");


logger.Info("SATISFACTION");
SatisfactionPrediction(logger);

Console.ReadLine();

static void SatisfactionPrediction(ILogger logger)
{
    var parser = new PersonSatisfactionDataParser();

    using var trainingReader = File.OpenText("Data/employees_train.txt");
    var trainingData = parser.Parse(trainingReader, out var scallingFunction);

    var (predictor, trainingPhases) = Predictor<PersonSatisfactionInput, PersonSatisfactionResult>.TrainFrom(trainingData, scallingFunction, new(maxEpoch: 1000));

    foreach (var (Epoch, Error, Accuracy) in trainingPhases)
        logger.Info($"iter = {Epoch,6}  error = {Error:F4} acc = {Accuracy:F4}");

    logger.Info($"Trained weights and bias: {string.Join(" ", predictor.Weights.Select(elem => elem.ToString("F4")))}");
    logger.Info("Accuracy of model on training data: " + predictor.Accuracy.ToString("F4"));


    var input = new PersonSatisfactionInput(false, 66, JobType.mgmt, 52100.00);

    logger.Info($"\nPredicting satisfaction for: {input}");

    var output = predictor.GetOutput(input, out var pValue);
    logger.Info($"Computed p-value = {pValue:F4}");
    logger.Info($"Predicted satisfaction = {output.Satisfaction}");
    logger.Info();

    using var testReader = File.OpenText("Data/employees_test.txt");
    var testData = parser.Parse(testReader, out _);

    var equalCounter = 0;

    foreach (var (testInput, testResult) in testData)
    {
        output = predictor.GetOutput(testInput, out _);

        var predicted = output.Satisfaction;
        var expected = testResult.Satisfaction;

        var isEqual = predicted == expected ? "==" : "!=";
        if (predicted == expected) equalCounter++;

        logger.Info($"{predicted} {isEqual} {expected} {testInput}");
    }

    logger.Info($"Actual accurancy {100.0 * equalCounter / testData.Count} %");
}


static void EmploymentPrediction(ILogger logger)
{
    var parser = new PersonEmploymentDataParser();

    using var trainingReader = File.OpenText("Data/employees_train.txt");
    var trainingData = parser.Parse(trainingReader, out var scallingFunction);

    var (predictor, trainingPhases) = Predictor<PersonEmploymentInput, PersonEmploymentResult>.TrainFrom(trainingData, scallingFunction, new(maxEpoch: 100));

    foreach (var (Epoch, Error, Accuracy) in trainingPhases)
        logger.Info($"iter = {Epoch,6}  error = {Error:F4} acc = {Accuracy:F4}");

    logger.Info($"Trained weights and bias: {string.Join(" ", predictor.Weights.Select(elem => elem.ToString("F4")))}");
    logger.Info("Accuracy of model on training data: " + predictor.Accuracy.ToString("F4"));


    var input = new PersonEmploymentInput(36, JobType.tech, 52000, Satisfaction.medium);
    logger.Info($"\nPredicting employment for: {input}");

    var output = predictor.GetOutput(input, out var pValue);

    static string GetEmployment(PersonEmploymentResult result) => result.IsContractor ? "Contractor" : "SalaryWorker";

    logger.Info($"Computed p-value = {pValue:F4}");
    logger.Info($"Predicted employment = {GetEmployment(output)}");
    logger.Info();



    using var testReader = File.OpenText("Data/employees_test.txt");
    var testData = parser.Parse(testReader, out _);

    var equalCounter = 0;

    foreach (var (testInput, testResult) in testData)
    {
        output = predictor.GetOutput(testInput, out _);

        var predicted = GetEmployment(output);
        var expected = GetEmployment(testResult);

        var isEqual = predicted == expected ? "==" : "!=";
        if (predicted == expected) equalCounter++;

        logger.Info($"{predicted} {isEqual} {expected} {testInput}");
    }

    logger.Info($"Actual accurancy {100.0 * equalCounter / testData.Count} %");
}

