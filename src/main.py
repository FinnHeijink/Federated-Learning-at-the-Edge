import torch
import torch.optim as optim

import Checkpointer
import Model
import Config
import Dataset
import ImageAugmenter
import Util

# Todo: implement CheckPointer & BYOL+Classifier splitted saving

def TrainBYOLEpoch(byol, device, dataset, optimizer, augmenter, checkpointer, epoch, maxEpochs):
    byol.train()  # Enables dropout

    print(f"Training BYOL Epoch {epoch + 1}: lr={optimizer.param_groups[0]['lr']}")

    maxTrainBatches = dataset.trainBatchCount() / dataset.batchSize
    for batchIndex, (data, target) in enumerate(dataset.trainingEnumeration()):
        dataView1, dataView2 = augmenter.createImagePairBatchSingleAugment(data)
        dataView1, dataView2, target = dataView1.to(device), dataView2.to(device), target.to(device)

        optimizer.zero_grad()
        loss = byol(dataView1, dataView2, target)
        loss.backward()
        optimizer.step()
        byol.stepEMA()

        # checkpointer.update(byol, optimizer, epoch, maxEpochs, batchIndex, maxTrainBatches)

        # Todo: let the checkpointer or so show this, or at least allow for configuration
        if batchIndex % 10 == 0:
            print(f"Epoch {epoch + 1}, batch {batchIndex}/{batchIndex / maxTrainBatches * 100:.1f}%: BYOLLoss={loss:.4f}")

def TrainClassifierEpoch(classifier, device, dataset, optimizer, checkpointer, epoch, maxEpochs):
    classifier.train()

    print(f"Training Classifier Epoch {epoch + 1}: lr={optimizer.param_groups[0]['lr']}")

    maxClassifierBatches = dataset.classificationBatchCount() / dataset.batchSize
    for batchIndex, (data, target) in enumerate(dataset.classificationEnumeration()):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = classifier.loss(data, target)
        loss.backward()
        optimizer.step()

        if batchIndex % 1 == 0:
            print(
                f"Epoch {epoch + 1}, batch {batchIndex}/{batchIndex / maxClassifierBatches * 100:.1f}%: classificationLoss={loss:.2f}")

def TestEpoch(classifier, device, dataset):
    classifier.eval() # Disable dropout

    testLoss = 0
    accuracy = 0
    with torch.no_grad():
        for batchIndex, (data, target) in enumerate(dataset.testingEnumeration()):
            data, target = data.to(device), target.to(device)
            loss, output, prediction = classifier.predictionLoss(data, target)

            testLoss += loss.item()
            accuracy += prediction.eq(target.view_as(prediction)).sum().item() / len(data)

    maxBatches = dataset.testBatchCount() / dataset.batchSize

    testLoss /= maxBatches
    accuracy /= maxBatches

    print(f"Evaluation: loss={testLoss:2f}, accuracy={accuracy * 100:.1f}%")

def main():
    mode = "pretrain" # Todo: get from cmdline args

    config = Config.GetConfig()

    torch.manual_seed(0)
    device = torch.device(config["device"])

    dataset = Dataset.Dataset(**config["dataset"])
    byol = Model.BYOL(**config["EMA"], **config["BYOL"]).to(device)
    classifier = Model.Classifier(**config["classifier"]).to(device)
    checkpointer = Checkpointer.Checkpointer(**config["checkpointer"])

    # Todo: scheduler

    if mode == "pretrain":
        byolOptimizer = getattr(optim, config["optimizer"]["name"])(byol.trainableParameters(), **config["optimizer"]["settings"])
        classifierOptimizer = getattr(optim, config["optimizer"]["name"])(classifier.trainableParameters(), **config["optimizer"]["settings"])

        # checkpointer.loadLastCheckpoint(byol, classifier, byolOptimizer, classifierOptimizer)
        augmenter = ImageAugmenter.ImageAugmenter(**config["augmenter"])

        for epoch in range(0, config["training"]["epochs"]):
            TrainBYOLEpoch(byol, device, dataset, byolOptimizer, augmenter, checkpointer, epoch, config["training"]["epochs"])
            classifier.copyEncoderFromBYOL(byol)
            TrainClassifierEpoch(classifier, device, dataset, classifierOptimizer, checkpointer, epoch, config["training"]["epochs"])
            if config["training"]["evaluateEveryEpoch"]:
                TestEpoch(classifier, device, dataset)
    elif mode == "eval":
        checkpointer.loadLastCheckpoint(byol, classifier, None, None)
        classifier.copyEncoderFromBYOL(byol)
        TestEpoch(classifier, device, dataset)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()