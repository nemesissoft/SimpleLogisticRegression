using Logit;

ILogger logger = new ConsoleLogger();

EmploymentPrediction(logger);

logger.Info("\n\n\n\n\n\n");

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


    //var input = new PersonSatisfactionInput(true, 36, JobType.tech, 52000);
    var input = new PersonSatisfactionInput(false, 66, JobType.mgmt, 52100.00);
        

    logger.Info($"\nPredicting satisfaction for: {input}");

    double p = predictor.GetOutput(input);
    logger.Info($"Computed p-value = {p:F4}");
    logger.Info($"Predicted satisfaction = {p switch
    {
        < 0.33 => Satisfaction.low,
        > 0.66 => Satisfaction.high,
        _ => Satisfaction.medium,
    }}");
    logger.Info();

    /*using var testReader = File.OpenText("Data/employees_test.txt");
    var testData = parser.Parse(testReader, out _);

    var equalCounter = 0;

    foreach (var (testInput, testResult) in testData)
    {
        double prob = predictor.GetOutput(testInput);
        var predicted = prob < 0.5 ? "SalaryWorker" : "Contractor";
        var expected = !testResult.IsContractor ? "SalaryWorker" : "Contractor";
        var isEqual = predicted == expected ? "==" : "!=";
        if (predicted == expected) equalCounter++;

        logger.Info($"{predicted} {isEqual} {expected} {testInput}");
    }

    logger.Info($"Actual accurancy {100.0 * equalCounter / testData.Count} %");*/
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

    double p = predictor.GetOutput(input);
    logger.Info($"Computed p-value = {p:F4}");
    logger.Info($"Predicted employment = {(p < 0.5 ? "SalaryWorker" : "Contractor")}");
    logger.Info();



    using var testReader = File.OpenText("Data/employees_test.txt");
    var testData = parser.Parse(testReader, out _);

    var equalCounter = 0;

    foreach (var (testInput, testResult) in testData)
    {
        double prob = predictor.GetOutput(testInput);
        var predicted = prob < 0.5 ? "SalaryWorker" : "Contractor";
        var expected = !testResult.IsContractor ? "SalaryWorker" : "Contractor";
        var isEqual = predicted == expected ? "==" : "!=";
        if (predicted == expected) equalCounter++;

        logger.Info($"{predicted} {isEqual} {expected} {testInput}");
    }

    logger.Info($"Actual accurancy {100.0 * equalCounter / testData.Count} %");
}

