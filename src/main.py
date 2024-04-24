import torch
import torch.optim as optim

import Checkpointer
import Model
import Config
import Dataset
import ImageAugmentation

def TrainEpoch(model, device, dataset, optimizer, checkpointer, epoch, maxEpochs):
    model.train()  # Enables dropout

    print(f"Epoch {epoch + 1}: lr={optimizer.param_groups[0]['lr']}")

    maxBatches = dataset.trainBatchCount()

    for batchIndex, (data, target) in enumerate(Dataset.trainingEnumeration()):
        data = ImageAugmentation.CreateImagePairBatch(data)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()

        checkpointer.update(model, optimizer, epoch, maxEpochs, batchIndex, maxBatches)

        #Todo: let the checkpointer or so show this, or at least allow for configuration
        if batchIndex % 10 == 0:
            print(
                f"Epoch {epoch + 1}, batch {batchIndex}/{batchIndex / maxBatches * 100:.1f}%: loss={loss:.2f}")

def TestEpoch(model, device, dataset):
    pass

def main():
    mode = "pretrain" #Todo: get from cmdline args

    config = Config.GetConfig()

    torch.manual_seed(0)
    device = torch.device(config.device)

    dataset = Dataset.Dataset('MNIST', **config.dataset)
    model = Model.BYOL(**config.model).to(device)
    checkpointer = Checkpointer.Checkpointer(**config.checkpointer)

    # Todo: scheduler

    if mode == "pretrain":
        optimizer = getattr(optim, config.optimizer.name)(model.trainableParameters(), **config.optimizer.settings)
        checkpointer.loadLastCheckpoint(model, optimizer)

        for epoch in range(0, config.training.epochs):
            TrainEpoch(model, device, dataset, optimizer, checkpointer, epoch, config.training.epochs)
            if config.training.evaluateEveryEpoch:
                TestEpoch(model, device, dataset)

    elif mode == "eval":
        checkpointer.loadLastCheckpoint(model, None)
        TestEpoch(model, device, dataset)
    else:
        raise NotImplementedError

if __name__=="__main__":
    main()