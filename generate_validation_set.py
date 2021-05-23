import os
import math
from pathlib import Path
import random

trainPath = Path('./assets/dataset/train')
validationPath = Path('./assets/dataset/validate')
os.mkdir(validationPath)
for foldeName in [f for f in os.listdir(trainPath)]:
    folder = trainPath / foldeName
    files = [f for f in os.listdir(folder)  if os.path.isfile(folder / f)]
    nFiles = len(files)
    selectedFiles = math.ceil(nFiles * 0.20)
    validationFiles = random.sample(files, selectedFiles)
    os.mkdir(validationPath / foldeName)
    for validationFile in validationFiles:
        os.rename(trainPath / foldeName / validationFile, validationPath / foldeName / validationFile)
    print(f"{foldeName} done")