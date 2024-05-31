import MainWrapped

defaultCmdline = ["--printStatistics=false", "--dataset.classificationSplit=0.1", "--EMA.enableSchedule=True", "--useHalfPrecision=false"]

useWrapper = True

def TestPerformance(encoderName, datasetName):
    cmdlineBYOL = defaultCmdline.copy()
    cmdlineBYOL.append("--mode=pretrain")
    cmdlineBYOL.append("--BYOL.encoderName=" + encoderName)
    cmdlineBYOL.append("--dataset.datasetName=" + datasetName)
    cmdlineBYOL.append("--training.epochs=10")
    cmdlineBYOL.append("--training.evaluateEveryNEpochs=0")
    cmdlineBYOL.append("--training.warmupEpochs=2")
    cmdlineBYOL.append("--training.classifierEpochs=0")
    cmdlineBYOL.append("--loadFromCheckpoint=false")

    byolOutput, _ = MainWrapped.RunMain(cmdlineBYOL, useWrapper)

    cmdlineBYOLClassifier = defaultCmdline.copy()
    cmdlineBYOLClassifier.append("--mode=evaltrainfrombyol")
    cmdlineBYOLClassifier.append("--BYOL.encoderName=" + encoderName)
    cmdlineBYOLClassifier.append("--dataset.datasetName=" + datasetName)
    cmdlineBYOLClassifier.append("--training.classifierEpochs=10")
    cmdlineBYOLClassifier.append("--training.evaluateEveryNEpochs=1")
    cmdlineBYOLClassifier.append("--training.warmupEpochs=2")
    cmdlineBYOLClassifier.append("--loadFromCheckpoint=true")

    byolClassifierOutput, byolAccuracy = MainWrapped.RunMain(cmdlineBYOLClassifier, useWrapper)

    cmdlineRandomBYOL = defaultCmdline.copy()
    cmdlineRandomBYOL.append("--mode=evaltrain")
    cmdlineRandomBYOL.append("--BYOL.encoderName=" + encoderName)
    cmdlineRandomBYOL.append("--dataset.datasetName=" + datasetName)
    cmdlineRandomBYOL.append("--training.classifierEpochs=10")
    cmdlineRandomBYOL.append("--training.evaluateEveryNEpochs=1")
    cmdlineRandomBYOL.append("--training.warmupEpochs=2")
    cmdlineRandomBYOL.append("--loadFromCheckpoint=false")

    randomBYOLClassifierOutput, randomBYOLAccuracy = MainWrapped.RunMain(cmdlineRandomBYOL, useWrapper)

    cmdlineNonBYOL = defaultCmdline.copy()
    cmdlineNonBYOL.append("--mode=classtrain")
    cmdlineNonBYOL.append("--BYOL.encoderName=" + encoderName)
    cmdlineNonBYOL.append("--dataset.datasetName=" + datasetName)
    cmdlineNonBYOL.append("--training.classifierEpochs=10")
    cmdlineNonBYOL.append("--training.evaluateEveryNEpochs=1")
    cmdlineNonBYOL.append("--training.warmupEpochs=2")
    cmdlineNonBYOL.append("--loadFromCheckpoint=false")

    nonBYOLClassifierOutput, nonBYOLAccuracy = MainWrapped.RunMain(cmdlineNonBYOL, useWrapper)

    f = open("PerformanceLog_" + encoderName + "_" + datasetName + ".txt", "wb")
    f.write("BYOL Output--------------------\n".encode('utf-8'))
    f.write(byolOutput)
    f.write("BYOL Output--------------------\n".encode('utf-8'))
    f.write(byolClassifierOutput)
    f.write("Random BYOL Output--------------------\n".encode('utf-8'))
    f.write(randomBYOLClassifierOutput)
    f.write("Supervised BYOL Output--------------------\n".encode('utf-8'))
    f.write(nonBYOLClassifierOutput)
    f.close()

    print(f"Encoder {encoderName}, dataset {datasetName}: BYOL {byolAccuracy * 100:.2f}%, random BYOL {randomBYOLAccuracy * 100:.2f}%, w/o BYOL {nonBYOLAccuracy * 100:.2f}%, improvement {(byolAccuracy - nonBYOLAccuracy) * 100:.2f}%")

    return byolAccuracy, nonBYOLAccuracy

TestPerformance("EncoderType2", "MNIST")