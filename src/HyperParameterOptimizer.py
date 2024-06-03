import skopt
import skopt.space as space
import MainWrapped
import pickle

searchSpace = list()
searchSpace.append(space.Real(1e-6, 1e-2, "log-uniform", name="--optimizer.settings.weight_decay"))
searchSpace.append(space.Real(0.5, 0.99, "log-uniform", name="--EMA.initialTau"))
#searchSpace.append(space.Integer(32, 512, "log-uniform", name="--BYOL.projector.hiddenSize"))
#searchSpace.append(space.Integer(16, 256, "log-uniform", name="--dataset.batchSize"))
#searchSpace.append(space.Categorical(["Encoder", "MobileNetV2Short", "MobileNetV2"], name="--BYOL.encoderName"))

@skopt.utils.use_named_args(searchSpace)
def evaluateModel(**params):
    cmdline = ["--mode=pretrain", "--BYOL.encoderName=Encoder", "--dataset.batchSize=128", "--BYOL.projector.hiddenSize=128", "--useHalfPrecision=False", "--quantization.enabled=True", "--loadFromCheckpoint=false", "--printStatistics=false", "--training.epochs=1", "--training.classifierEpochs=5", "--dataset.classificationSplit=0.1", "--training.evaluateEveryNEpochs=1", "--training.warmupEpochs=0", "--EMA.enableSchedule=False"]

    for name, value in params.items():
       cmdline.append(name + "=" + str(value))

    print("Running with cmdline:", cmdline)

    output, accuracy = MainWrapped.RunMain(cmdline, useWrapper=False)
    print("Accuracy:", accuracy)

    return 1.0 - accuracy

result = skopt.gp_minimize(evaluateModel, searchSpace)
print(result)

with open("HyperParameterOptimizerResult.dat", "wb") as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
