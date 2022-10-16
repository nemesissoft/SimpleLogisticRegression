namespace Logit;

interface ILogger
{
    void Info(string message = "");

    void Info(FormattableString message) => Info(FormattableString.Invariant(message));
}
class ConsoleLogger : ILogger
{
    public void Info(string message = "") => Console.WriteLine(message);
}