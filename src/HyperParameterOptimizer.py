import skopt
import skopt.space as space
import subprocess

searchSpace = list()
searchSpace.append(space.Real(1e-6, 1e-2, "log-uniform", name="--optimizer.settings.weight_decay"))
searchSpace.append(space.Real(0.5, 0.99, "log-uniform", name="--EMA.initialTau"))
searchSpace.append(space.Integer(32, 512, "log-uniform", name="--BYOL.projector.hiddenSize"))
searchSpace.append(space.Integer(16, 256, "log-uniform", name="--dataset.batchSize"))
searchSpace.append(space.Categorical(["Encoder", "MobileNetV2Short", "MobileNetV2"], name="--BYOL.encoderName"))

@skopt.utils.use_named_args(searchSpace)
def evaluateModel(**params):
    cmdline = ["src\\RunMainWrapped.bat", "--mode=pretrain", "--loadFromCheckpoint=false", "--printStatistics=false", "--training.epochs=1", "--training.classifierEpochs=5", "--dataset.classificationSplit=0.1", "--training.evaluateEveryNEpochs=1", "--training.warmupEpochs=0", "--EMA.enableSchedule=False"]

    for name, value in params.items():
       cmdline.append(name + "=" + str(value))

    print("Running with cmdline:", cmdline)

    result = subprocess.run(cmdline, stdout=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError

    output = str(result.stdout)
    print(output)

    accuracy = output[output.find("Final accuracy: ") + len("Final accuracy: "):]
    accuracy = accuracy[:accuracy.find("\\")]
    accuracy = float(accuracy)

    print("Accuracy:", accuracy)

    return 1.0 - accuracy

result = skopt.gp_minimize(evaluateModel, searchSpace)
print(result)