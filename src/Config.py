import Checkpointer

def GetConfig():
    return dict(
        device="cuda",
        imageDims=(28, 28),

        training=dict(
            epochs=40,
            evaluateEveryEpoch=True,
        ),
        dataset=dict(
            datasetName="MNIST",
            batchSize=64
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
                imageDims=(28, 28),
                imageChannels=1
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