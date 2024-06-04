import torch
import torch.optim as optim
from torchsummary import summary

import argparse

import Checkpointer
import Model
import Config
import Dataset
import ImageAugmenter
import Util

def TrainBYOLEpoch(byol, device, dataset, optimizer, augmenter, checkpointer, epoch, maxEpochs):
    byol.train()  # Enables dropout

    print(f"Training BYOL Epoch {epoch + 1}: lr={optimizer.param_groups[0]['lr']}, tau={byol.emaScheduler.getTau()}")

    maxTrainBatches = dataset.trainBatchCount() / dataset.batchSize
    for batchIndex, (data, target) in enumerate(dataset.trainingEnumeration()):
        data = data.to(device)
        dataView1, dataView2 = augmenter.createImagePairBatch(data)
        #dataView1, dataView2 = dataView1.to(device), dataView2.to(device)

        optimizer.zero_grad()
        loss = byol(dataView1, dataView2)
        loss.backward()
        optimizer.step()
        byol.stepEMA()

        checkpointer.update(byol, optimizer, epoch, maxEpochs, batchIndex, maxTrainBatches)

        # Todo: let the checkpointer or so show this, or at least allow for configuration
        if batchIndex % 10 == 0:
            print(f"Epoch {epoch + 1}, batch {batchIndex}/{batchIndex / maxTrainBatches * 100:.1f}%: BYOLLoss={loss:.4f}")

def TrainClassifierEpoch(classifier, device, dataset, optimizer, checkpointer, epoch, maxEpochs, useHalfPrecision):
    classifier.train()

    print(f"Training Classifier Epoch {epoch + 1}: lr={optimizer.param_groups[0]['lr']}")

    maxClassifierBatches = dataset.classificationBatchCount() / dataset.batchSize
    for batchIndex, (data, target) in enumerate(dataset.classificationEnumeration()):
        data, target = data.to(device), target.to(device)

        if useHalfPrecision:
            data = data.half()

        optimizer.zero_grad()
        loss = classifier.loss(data, target)
        loss.backward()
        optimizer.step()

        checkpointer.update(classifier, optimizer, epoch, maxEpochs, batchIndex, maxClassifierBatches)

        if batchIndex % 10 == 0:
            print(
                f"Epoch {epoch + 1}, batch {batchIndex}/{batchIndex / maxClassifierBatches * 100:.1f}%: classificationLoss={loss:.2f}")


def TestEpoch(classifier, device, dataset, useHalfPrecision):
    classifier.eval() # Disable dropout

    testLoss = 0
    accuracy = 0
    batchCount = 0
    with torch.no_grad():
        for batchIndex, (data, target) in enumerate(dataset.testingEnumeration()):
            data, target = data.to(device), target.to(device)

            if useHalfPrecision:
                data = data.half()

            loss, output, prediction = classifier.predictionLoss(data, target)

            testLoss += loss.item()
            accuracy += prediction.eq(target.view_as(prediction)).sum().item() / len(data)

            batchCount += 1

    testLoss /= batchCount
    accuracy /= batchCount

    print(f"Evaluation: loss={testLoss:2f}, accuracy={accuracy * 100:.1f}%")

    return testLoss, accuracy

def ParseArgs(config):
    parser = argparse.ArgumentParser()

    # From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            pass  # error condition maybe?

    def PopulateDict(prefix, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                PopulateDict(prefix + key + ".", value)
            else:
                if type(value) == bool:
                    parser.add_argument("--" + prefix + key, required=False, type=t_or_f)
                else:
                    parser.add_argument("--" + prefix + key, required=False, type=type(value))

    PopulateDict("", config)
    result = parser.parse_args()

    def RetrieveDict(prefix, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                RetrieveDict(prefix + key + ".", value)
            else:
                resultValue = getattr(result, prefix + key)
                if not (resultValue is None):
                    dictionary[key] = resultValue

    RetrieveDict("", config)

def main():
    config = Config.GetConfig(doPostConfig=False)
    ParseArgs(config)
    Config.DoPostConfig(config)

    torch.manual_seed(0)
    device = Util.GetDeviceFromConfig(config)
    statistics = []

    dataset = Dataset.Dataset(**config["dataset"])
    emaScheduler = Util.EMAScheduler(**config["EMA"])
    byol = Model.BYOL(emaScheduler, **config["BYOL"]).to(device)
    classifier = Model.Classifier(**config["classifier"]).to(device)
    byolCheckpointer = Checkpointer.Checkpointer(**config["checkpointer"], prefix="BYOL")
    classifierCheckpointer = Checkpointer.Checkpointer(**config["checkpointer"], prefix="Classifier")

    Model.SetUseReLU1(config["useReLU1"])

    if config["mode"] == "pretrain":
        byolOptimizer = getattr(optim, config["optimizer"]["name"])(byol.trainableParameters(), **config["optimizer"]["settings"])
        classifierOptimizer = getattr(optim, config["optimizer"]["name"])(classifier.trainableParameters(), **config["optimizer"]["settings"])

        startEpoch = 0
        if config["loadFromCheckpoint"]:
            startEpoch = byolCheckpointer.loadCheckpoint(config["loadFromSpecificCheckpoint"], byol, byolOptimizer)
            classifierCheckpointer.loadCheckpoint(config["loadFromSpecificCheckpoint"], classifier, classifierOptimizer)

        lrScheduler = Util.WarmupCosineScheduler(byolOptimizer, startEpoch, config["training"]["epochs"], config["training"]["warmupEpochs"], config["optimizer"]["settings"]["lr"])
        emaScheduler.startStep(startEpoch)

        augmenter = ImageAugmenter.ImageAugmenter(**config["augmenter"])

        try:
            for epoch in range(startEpoch, config["training"]["epochs"]):
                TrainBYOLEpoch(byol, device, dataset, byolOptimizer, augmenter, byolCheckpointer, epoch, config["training"]["epochs"])
                classifier.copyEncoderFromBYOL(byol)
                for i in range(config["training"]["classifierEpochs"]):
                    TrainClassifierEpoch(classifier, device, dataset, classifierOptimizer, classifierCheckpointer, epoch, config["training"]["epochs"], config["useHalfPrecision"])

                if config["training"]["evaluateEveryNEpochs"] != 0 and epoch % config["training"]["evaluateEveryNEpochs"] == 0:
                    testResults = TestEpoch(classifier, device, dataset, config["useHalfPrecision"])
                    statistics.append((*testResults, epoch))

                lrScheduler.step()
                classifierOptimizer.param_groups[0]["lr"] = byolOptimizer.param_groups[0]["lr"]
                emaScheduler.step(epoch)
        except KeyboardInterrupt:
            pass

        for i in range(config["training"]["finalclassifierEpochs"]):
            TrainClassifierEpoch(classifier, device, dataset, classifierOptimizer, classifierCheckpointer, config["training"]["epochs"], config["training"]["epochs"], config["useHalfPrecision"])
        
        testResults = TestEpoch(classifier, device, dataset, config["useHalfPrecision"])
        statistics.append((*testResults, config["training"]["epochs"]))

    elif config["mode"] == "eval":
        if config["loadFromCheckpoint"]:
            classifierCheckpointer.loadCheckpoint(config["loadFromSpecificCheckpoint"], classifier, None)

        TestEpoch(classifier, device, dataset, config["useHalfPrecision"])
    elif config["mode"] == "evaltrain" or config["mode"] == "evaltrainfrombyol":
        classifierOptimizer = getattr(optim, config["optimizer"]["name"])(classifier.trainableParameters(), **config["optimizer"]["settings"])

        startEpoch = 0

        if config["mode"] == "evaltrainfrombyol":
            if config["loadFromCheckpoint"]:
                startEpoch = byolCheckpointer.loadCheckpoint(config["loadFromSpecificCheckpoint"], byol, None)
            classifier.copyEncoderFromBYOL(byol)
        else:
            if config["loadFromCheckpoint"]:
                startEpoch = classifierCheckpointer.loadCheckpoint(config["loadFromSpecificCheckpoint"], classifier, classifierOptimizer)

        lrScheduler = Util.WarmupCosineScheduler(classifierOptimizer, startEpoch, config["training"]["epochs"], config["training"]["warmupEpochs"], config["optimizer"]["settings"]["lr"])

        try:
            for epoch in range(startEpoch, config["training"]["classifierEpochs"]):
                TrainClassifierEpoch(classifier, device, dataset, classifierOptimizer, classifierCheckpointer, epoch, config["training"]["classifierEpochs"], config["useHalfPrecision"])

                if config["training"]["evaluateEveryNEpochs"] != 0 and epoch % config["training"]["evaluateEveryNEpochs"] == 0:
                    testResults = TestEpoch(classifier, device, dataset, config["useHalfPrecision"])
                    statistics.append((*testResults, epoch))

            lrScheduler.step()
        except KeyboardInterrupt:
            pass
    elif config["mode"] == "classtrain":
        classifier.setAllowTrainingEncoder() #Before classifier.trainableParameters()
        classifierOptimizer = getattr(optim, config["optimizer"]["name"])(classifier.trainableParameters(), **config["optimizer"]["settings"])

        startEpoch = 0
        lrScheduler = Util.WarmupCosineScheduler(classifierOptimizer, startEpoch, config["training"]["epochs"], config["training"]["warmupEpochs"], config["optimizer"]["settings"]["lr"])

        try:
            for epoch in range(startEpoch, config["training"]["classifierEpochs"]):
                TrainClassifierEpoch(classifier, device, dataset, classifierOptimizer, classifierCheckpointer, epoch, config["training"]["classifierEpochs"], config["useHalfPrecision"])

                if config["training"]["evaluateEveryNEpochs"] != 0 and epoch % config["training"]["evaluateEveryNEpochs"] == 0:
                    testResults = TestEpoch(classifier, device, dataset, config["useHalfPrecision"])
                    statistics.append((*testResults, epoch))

            lrScheduler.step()
        except KeyboardInterrupt:
            pass
    else:
        raise NotImplementedError

    if config["printStatistics"]:
        computeCost = byol.getForwardComputeCost()
        print("Multiplies:", computeCost[0])
        print("Adds:", computeCost[1])
        print("Memory:", computeCost[2] * (2 if config["useHalfPrecision"] else 4))

    if len(statistics):
        if config["printStatistics"]:
            Util.PlotStatistics(statistics)

        print("Final accuracy:", statistics[-1][1])

if __name__ == "__main__":
    main()