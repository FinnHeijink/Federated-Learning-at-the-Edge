import MainWrapped
import numpy as np
import matplotlib.pyplot as plt

defaultCmdline = ["src\\RunMainWrapped.bat", "--mode=pretrain", "--loadFromCheckpoint=false", "--printStatistics=false",
           "--training.epochs=50", "--training.classifierEpochs=1", "--dataset.classificationSplit=0.1",
           "--training.evaluateEveryNEpochs=5", "--training.warmupEpochs=5", "--EMA.enableSchedule=True"]

def PlotAccuracyVSTau():
    def Evaluate(tau):
        cmdline = defaultCmdline.copy()
        cmdline.append("--EMA.initialTau=" + str(tau))
        output, accuracy = MainWrapped.RunMain(cmdline)
        return accuracy

    taus = np.logspace(0.5, 0.99, 4)
    accuracies = np.array([Evaluate(tau) for tau in taus])

    print(taus, accuracies)
    plt.figure()
    plt.plot(taus, accuracies * 100)
    plt.xlabel("$\\tau$")
    plt.ylabel("Accuracy [%]")
    plt.savefig("AccuracyVsTau.svg")
    plt.grid()
    plt.show()

PlotAccuracyVSTau()