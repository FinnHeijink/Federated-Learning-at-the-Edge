import Checkpointer

def GetConfig():
    config = dict(
        device="cuda",

        augmenter=dict(
            imageDims=(0, 0), #autoset
            applyColorAugments=False,
            applyFlips=False,
        ),

        training=dict(
            epochs=40,
            evaluateEveryEpoch=True,
        ),
        dataset=dict(
            datasetName="CIFAR10",
            normalization=None, #autoset
            batchSize=64,
            classificationSplit=0.01,
        ),
        EMA=dict(
            initialTau=0.95
        ),
        model=dict(
            classCount=10,
            projector=dict(
                hiddenSize=32,
                outputSize=10,
            ),
            predictor=dict(
                hiddenSize=32
            ),
            classifier=dict(
                hiddenSize=10
            ),
            encoder=dict(
                imageDims=(0, 0), #autoset
                imageChannels=0 #autoset
            ),
            batchNorm=dict(
                eps=1e-5,
                momentum=0.1
            )
        ),
        optimizer=dict(
            name="Adam",
            settings=dict(
                lr=0.001
            )
        ),
        checkpointer=dict(
            directory="src/checkpoints",
            #checkpointMode=Checkpointer.CheckpointMode.EVERY_N_SECS,
            checkpointMode=Checkpointer.CheckpointMode.EVERY_EPOCH,
            checkPointEveryNSecs=30
        )
    )

    if config["dataset"]["datasetName"] == "MNIST":
        config["augmenter"]["imageDims"] = (28, 28)
        config["dataset"]["normalization"] = ((0.1307,), (0.3081,))
        config["model"]["encoder"]["imageDims"] = (28, 28)
        config["model"]["encoder"]["imageChannels"] = 1
    elif config["dataset"]["datasetName"] == "CIFAR10":
        config["augmenter"]["imageDims"] = (32, 32)
        config["augmenter"]["applyColorAugments"] = True
        config["augmenter"]["applyFlips"] = True
        config["dataset"]["normalization"] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        config["model"]["encoder"]["imageDims"] = (32, 32)
        config["model"]["encoder"]["imageChannels"] = 3
    else:
        raise NotImplementedError

    return config