import Checkpointer

def GetConfig(doPostConfig=True):
    config = dict(
        device="cuda",
        mode="pretrain",
        loadFromCheckpoint=False,
        loadFromSpecificCheckpoint=None,
        printStatistics=True,
        useHalfPrecision=True,
        useReLU1=True,

        augmenter=dict(
            imageDims=(0, 0), #autoset
            applyColorAugments=False,
            applyFlips=False,
        ),

        training=dict(
            epochs=50,
            warmupEpochs=1,
            evaluateEveryNEpochs=1,
            classifierEpochs=1,
            finalclassifierEpochs=20,
        ),
        dataset=dict(
            datasetName="KMNIST",
            normalization=None, #autoset
            batchSize=64,
            classificationSplit=0.1,
        ),
        EMA=dict(
            initialTau=0.0,
            epochCount=None, #autoset
            enableSchedule=True
        ),
        classifier=dict(
            hiddenSize=128,
            batchNorm=None #autoset
        ),
        BYOL=dict(
            encoderName="EncoderType4",
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
        #optimizer=dict(
        #    name="AdamW",
        #    settings=dict(
        #        lr=0.0003,
        #        weight_decay=0.0001
        #    )
        #),
        optimizer=dict(
            name="SGD",
            settings=dict(
                lr=0.05,
                weight_decay=0.0001,
                momentum=0.9
            )
        ),
        batchNorm=dict(
            eps=1e-5,
            momentum=0.1
        ),
        quantization = dict(
            enabled=True,
            nb=12,
            nf=7,
            quantizeWeights=False,
            useCustomConv=True
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

    if doPostConfig:
        DoPostConfig(config)

    return config

def DoPostConfig(config):
    config["BYOL"]["batchNorm"] = config["batchNorm"]
    config["BYOL"]["dtypeName"] = "float16" if config["useHalfPrecision"] else "float32"
    config["BYOL"]["quantization"] = config["quantization"]
    config["classifier"]["dtypeName"] = "float16" if config["useHalfPrecision"] else "float32"
    config["classifier"]["batchNorm"] = config["batchNorm"]
    config["classifier"]["encoder"] = config["BYOL"]["encoder"]
    config["classifier"]["encoderName"] = config["BYOL"]["encoderName"]
    config["classifier"]["quantization"] = config["quantization"]
    config["dataBuffer"]["batchSize"] = config["dataset"]["batchSize"]
    config["EMA"]["epochCount"] = config["training"]["epochs"]
    config["augmenter"]["useHalfPrecision"] = config["useHalfPrecision"]

    config["optimizer"]["settings"]["lr"] = config["optimizer"]["settings"]["lr"] * config["dataset"]["batchSize"] / 64

    if config["EMA"]["initialTau"] > 0.01:  # Tau=0 means EMA disabled, so don't scale it. Otherwise, do scale.
        config["EMA"]["initialTau"] = 1 - (1 - config["EMA"]["initialTau"]) * (config["dataset"]["batchSize"] / 128)

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
    elif config["dataset"]["datasetName"] == "FashionMNIST":
        config["augmenter"]["imageDims"] = (28, 28)
        config["dataset"]["normalization"] = ((0.1307,), (0.3081,))
        config["BYOL"]["encoder"]["imageDims"] = (28, 28)
        config["BYOL"]["encoder"]["imageChannels"] = 1
        config["classifier"]["classCount"] = 10
    elif config["dataset"]["datasetName"] == "KMNIST":
        config["augmenter"]["imageDims"] = (28, 28)
        config["dataset"]["normalization"] = ((0.1917,), (0.3483,))
        config["BYOL"]["encoder"]["imageDims"] = (28, 28)
        config["BYOL"]["encoder"]["imageChannels"] = 1
        config["classifier"]["classCount"] = 10
    elif config["dataset"]["datasetName"] == "EMNIST":
        config["augmenter"]["imageDims"] = (28, 28)
        config["dataset"]["normalization"] = ((0.1307,), (0.3081,))
        config["BYOL"]["encoder"]["imageDims"] = (28, 28)
        config["BYOL"]["encoder"]["imageChannels"] = 1
        config["classifier"]["classCount"] = 47
    else:
        raise NotImplementedError
