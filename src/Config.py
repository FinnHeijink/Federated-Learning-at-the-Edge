import Checkpointer

def GetConfig():
    return dict(
        device="cpu",

        training=dict(
            epochs=40,
            evaluateEveryEpoch=True,
        ),
        dataset=dict(
            datasetName="MNIST",
            batchSize=64
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
                hiddenSize=32
            ),
            encoder=dict(
                imageDims=(28, 28),
                imageChannels=1
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
            checkpointMode=Checkpointer.CheckpointMode.EVERY_N_SECS,
            checkPointEveryNSecs=30
        )
    )