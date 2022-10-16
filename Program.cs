using Logit;

Employment();

Console.ReadLine();

static void Employment()
{
    ILogger logger = new ConsoleLogger();

    using var trainingReader = File.OpenText("Data/employees_train.txt");
    var trainingData = new PersonEmploymentDataParser().Parse(trainingReader, out var scallingFunction);

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
    var testData = new PersonEmploymentDataParser().Parse(testReader, out _);

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

