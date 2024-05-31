import subprocess

def RunMain(cmdline, useWrapper=False):

    if useWrapper:
        cmdline = ["src\\RunMainWrapped.bat"] + cmdline
    else:
        cmdline = ["python", "src\\main.py"] + cmdline

    result = subprocess.Popen(cmdline, stdout=subprocess.PIPE)

    output = ""
    for line in result.stdout:
        line = line.decode("utf-8")
        output += line
        print(line, end="")

    returncode = result.wait()
    if returncode != 0:
        print("Return code:", returncode)
        raise RuntimeError

    try:
        accuracy = output[output.find("Final accuracy: ") + len("Final accuracy: "):]
        accuracy = accuracy[:accuracy.find("\r")]
        accuracy = float(accuracy)
    except:
        accuracy = 0

    return output, accuracy
