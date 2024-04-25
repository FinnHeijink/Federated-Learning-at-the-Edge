import torch
import torch.optim as optim

import Checkpointer
import Model
import Config
import Dataset
import ImageAugmenter

def TrainEpoch(model, device, dataset, optimizer, augmenter, checkpointer, epoch, maxEpochs):
    model.train()  # Enables dropout

    print(f"Epoch {epoch + 1}: lr={optimizer.param_groups[0]['lr']}")

    maxBatches = dataset.trainBatchCount() / dataset.batchSize

    for batchIndex, (data, target) in enumerate(dataset.trainingEnumeration()):
        dataView1, dataView2 = augmenter.createImagePairBatch(data)
        dataView1, dataView2, target = dataView1.to(device), dataView2.to(device), target.to(device)

        optimizer.zero_grad()
        loss, classificationLoss, onlineLoss = model(dataView1, dataView2, target)
        loss.backward()
        optimizer.step()
        model.stepEMA()

        checkpointer.update(model, optimizer, epoch, maxEpochs, batchIndex, maxBatches)

        #Todo: let the checkpointer or so show this, or at least allow for configuration
        if batchIndex % 10 == 0:
            print(
                f"Epoch {epoch + 1}, batch {batchIndex}/{batchIndex / maxBatches * 100:.1f}%: loss={loss:.2f}, classificationLoss={classificationLoss.item():.2f}, onlineLoss={onlineLoss.item():.4f}")

def TestEpoch(model, device, dataset):
    model.eval() # Disable dropout

    testLoss = 0
    accuracy = 0
    with torch.no_grad():
        for batchIndex, (data, target) in enumerate(dataset.testingEnumeration()):
            data, target = data.to(device), target.to(device)
            output, prediction, loss = model.predictEval(data, target)

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
    ema = Model.EMA(**config["EMA"])
    model = Model.BYOL(ema, **config["model"]).to(device)
    checkpointer = Checkpointer.Checkpointer(**config["checkpointer"])

    # Todo: scheduler

    if mode == "pretrain":
        optimizer = getattr(optim, config["optimizer"]["name"])(model.trainableParameters(), **config["optimizer"]["settings"])
        checkpointer.loadLastCheckpoint(model, optimizer)
        augmenter = ImageAugmenter.ImageAugmenter(imageDims=config["imageDims"])

        for epoch in range(0, config["training"]["epochs"]):
            TrainEpoch(model, device, dataset, optimizer, augmenter, checkpointer, epoch, config["training"]["epochs"])
            if config["training"]["evaluateEveryEpoch"]:
                TestEpoch(model, device, dataset)

    elif mode == "eval":
        checkpointer.loadLastCheckpoint(model, None)
        TestEpoch(model, device, dataset)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()