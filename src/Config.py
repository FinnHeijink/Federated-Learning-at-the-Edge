import Checkpointer

def GetConfig():
    config = dict(
        device="cuda",
        mode="pretrain",
        loadFromCheckpoint=True,
        loadFromSpecificCheckpoint=None,
        printStatistics=True,

        augmenter=dict(
            imageDims=(0, 0), #autoset
            applyColorAugments=False,
            applyFlips=False,
        ),

        training=dict(
            epochs=4000,
            evaluateEveryNEpochs=1,
            classifierEpochs=1,
        ),
        dataset=dict(
            datasetName="CIFAR10",
            normalization=None, #autoset
            batchSize=32,
            classificationSplit=0.1,
        ),
        EMA=dict(
            initialTau=0.90
        ),
        classifier=dict(
            hiddenSize=128,
            batchNorm=None #autoset
        ),
        BYOL=dict(
            encoderName="MobileNetV2Short",
            projector=dict(
                hiddenSize=128,
                outputSize=32,
            ),
            predictor=dict(
                hiddenSize=128
            ),
            encoder=dict(
                imageDims=(0, 0), #autoset
                imageChannels=0 #autoset
            ),
            batchNorm=None #autoset
        ),
        optimizer=dict(
            name="AdamW",
            settings=dict(
                lr=0.0003,
                weight_decay=0.0001
            )
        ),
        #optimizer=dict(
        #    name="SGD",
        #    settings=dict(
        #        lr=0.01,
        #        weight_decay=4e-5,
        #        momentum=0.9
        #    )
        #),
        batchNorm=dict( # Todo: batchnorm config is not implemented in MobileNetV2
            eps=1e-5,
            momentum=0.1
        ),
        checkpointer=dict(
            directory="src/checkpoints",
            #checkpointMode=Checkpointer.CheckpointMode.EVERY_N_SECS,
            checkpointMode=Checkpointer.CheckpointMode.EVERY_EPOCH,
            checkPointEveryNSecs=30,
            saveOptimizerData=True
        ),
        dataBuffer=dict(
            datasetLoadBatchSize=16,
            bufferSize=128,
            batchSize=None, #autoset
            lazyScoringInterval=50,
            epochStreamCount=16,
        ),
        client=dict(
            serverSyncEveryNEpochs=100,
            updateBufferEveryNEpochs=1
        ),
        server=dict(
            classifierTrainEpochs=25,
        )
    )

    config["BYOL"]["batchNorm"] = config["batchNorm"]
    config["classifier"]["batchNorm"] = config["batchNorm"]
    config["classifier"]["encoder"] = config["BYOL"]["encoder"]
    config["classifier"]["encoderName"] = config["BYOL"]["encoderName"]
    config["dataBuffer"]["batchSize"] = config["dataset"]["batchSize"]

    if config["EMA"]["initialTau"] > 0.01: # Tau=0 means EMA disabled, so don't scale it.
        config["EMA"]["initialTau"] = 1 - (1 - config["EMA"]["initialTau"]) * (32 / config["dataset"]["batchSize"])

    if config["dataset"]["datasetName"] == "MNIST":
        config["augmenter"]["imageDims"] = (28, 28)
        config["dataset"]["normalization"] = ((0.1307,), (0.3081,))
        config["BYOL"]["encoder"]["imageDims"] = (28, 28)
        config["BYOL"]["encoder"]["imageChannels"] = 1
        config["classifier"]["classCount"] = 10
    elif config["dataset"]["datasetName"] == "CIFAR10":
        config["augmenter"]["imageDims"] = (32, 32)
        config["augmenter"]["applyColorAugments"] = True
        config["augmenter"]["applyFlips"] = True
        config["dataset"]["normalization"] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        config["BYOL"]["encoder"]["imageDims"] = (32, 32)
        config["BYOL"]["encoder"]["imageChannels"] = 3
        config["classifier"]["classCount"] = 10
    else:
        raise NotImplementedError

    return config