import subprocess
import sys
import os

sys.path.append(os.path.dirname(__file__))


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_requirements(requirements_file):
    with open(requirements_file, "r") as f:
        for line in f:
            package = line.strip()
            if package and not package.startswith("#"):
                try:
                    __import__(package.split("==")[0])
                except ImportError:
                    install(package)


if __name__ == "__main__":
    check_requirements("requirements.txt")
