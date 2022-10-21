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

    var trainingSets = PredictorFactory.TrainMultipleClassFrom(trainingData, scallingFunction, new(0.1, 100000));

    int numClass = 0;
    foreach (var (predictor, trainingPhases) in trainingSets)
    {
        logger.Info($"Training for class {numClass++}");
        foreach (var (Epoch, Error, Accuracy) in trainingPhases)
            logger.Info($"iter = {Epoch,6}  error = {Error:F4} acc = {Accuracy:F4}");
        logger.Info($"Trained weights and bias: {string.Join(" ", predictor.Weights.Select(elem => elem.ToString("F4")))}");
        logger.Info($"Accuracy of model on training data: {predictor.Accuracy:F4}");
    }


    var input = new PersonSatisfactionInput(false, 66, JobType.mgmt, 52100.00);
    logger.Info($"\nPredicting satisfaction for: {input}");

    var predictors = trainingSets.Select(x => x.Predictor);
    var decoder = PersonSatisfactionDecoder.Instance;

    var outputs = predictors.Select(p => p.GetOutput(input)).ToList();
    logger.Info($"Computed p-values = [{string.Join(" ", outputs.Select(elem => elem.ToString("F4")))}]");
    logger.Info($"Predicted satisfaction = {decoder.Decode(outputs).Satisfaction}");
    logger.Info();

    using var testReader = File.OpenText("Data/employees_test.txt");
    var testData = parser.Parse(testReader, out _);

    var equalCounter = 0;

    foreach (var (testInput, testResult) in testData)
    {
        outputs = predictors.Select(p => p.GetOutput(testInput)).ToList();

        var predicted = decoder.Decode(outputs).Satisfaction;
        var expected = testResult.Satisfaction;

        var isEqual = predicted == expected ? "==" : "!=";
        if (predicted == expected) equalCounter++;

        logger.Info($"{predicted,7} {isEqual} {expected,7} [{string.Join(" ", outputs.Select(elem => elem.ToString("F4")))}] {testInput.ToString().Replace(nameof(PersonSatisfactionInput), "")}");
    }

    logger.Info($"Actual accurancy {100.0 * equalCounter / testData.Count} %"); //MacroAccuracy when training set is used
}


static void EmploymentPrediction(ILogger logger)
{
    var parser = new PersonEmploymentDataParser();

    using var trainingReader = File.OpenText("Data/employees_train.txt");
    var trainingData = parser.Parse(trainingReader, out var scallingFunction);

    var (predictor, trainingPhases) = PredictorFactory.TrainFrom(trainingData, scallingFunction, new(maxEpoch: 100));

    foreach (var (Epoch, Error, Accuracy) in trainingPhases)
        logger.Info($"iter = {Epoch,6}  error = {Error:F4} acc = {Accuracy:F4}");

    logger.Info($"Trained weights and bias: {string.Join(" ", predictor.Weights.Select(elem => elem.ToString("F4")))}");
    logger.Info($"Accuracy of model on training data: {predictor.Accuracy:F4}");


    var input = new PersonEmploymentInput(36, JobType.tech, 52000, Satisfaction.medium);
    logger.Info($"\nPredicting employment for: {input}");

    var pValue = predictor.GetOutput(input);
    var decoder = EmploymentResultDecoder.Instance;

    static string GetEmployment(PersonEmploymentResult result) => result.IsContractor ? "Contractor" : "SalaryWorker";

    logger.Info($"Computed p-value = {pValue:F4}");
    logger.Info($"Predicted employment = {GetEmployment(decoder.Decode(pValue))}");
    logger.Info();



    using var testReader = File.OpenText("Data/employees_test.txt");
    var testData = parser.Parse(testReader, out _);

    var equalCounter = 0;

    foreach (var (testInput, testResult) in testData)
    {
        pValue = predictor.GetOutput(testInput);

        var predicted = GetEmployment(decoder.Decode(pValue));
        var expected = GetEmployment(testResult);

        var isEqual = predicted == expected ? "==" : "!=";
        if (predicted == expected) equalCounter++;

        logger.Info($"{predicted} {isEqual} {expected} {testInput}");
    }

    logger.Info($"Actual accurancy {100.0 * equalCounter / testData.Count} %");
}

