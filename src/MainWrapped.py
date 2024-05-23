import subprocess

def RunMain(cmdline):
    result = subprocess.run(cmdline, stdout=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError

    output = str(result.stdout)

    accuracy = output[output.find("Final accuracy: ") + len("Final accuracy: "):]
    accuracy = accuracy[:accuracy.find("\\")]
    accuracy = float(accuracy)

    return output, accuracy
