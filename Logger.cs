using System;

namespace Logit;

interface ILogger
{
    void Info(string message);
}
class ConsoleLogger : ILogger
{
    public void Info(string message) => Console.WriteLine(message);
}