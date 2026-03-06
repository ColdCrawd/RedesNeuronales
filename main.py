import json
import numpy as np


from CapaDensa import CapaDensa

import sys
import matplotlib.pyplot as plt

from RedNeuronal import RedNeuronal
from utils import cargar_capas, cargar_entrada

red_path = sys.argv[1]
data_path = sys.argv[2]

#"./mnist_dataset/mnist_mlp_pretty.json"
#"./mnist_dataset/mnist_custom_ds.npz"
red = RedNeuronal(cargar_capas(red_path))
input, labels = cargar_entrada(data_path)

incorrectas = []
presiciones = 0
for imagen, label in zip(input, labels):
    red.forward(imagen.reshape(1, -1))
    presiciones += red.precision(label)

    if label != red.prediccion():
        incorrectas.append(imagen)

print( presiciones /len(labels))


"""
input, labels = cargar_entrada("./mnist_dataset/mnist_custom_ds.npz")
capas = cargar_capas("./mnist_dataset/mnist_mlp_pretty.json")
input_2 = [input[2]]
capas[0].forward(input)
#print(capas[0].output)
capas[1].forward(capas[0].output)
print(capas[1].output)
for value in capas[1].output:
    print(np.max(value), np.argmax(value))

print("OUTPUT\n", capas[1].output)
print(capas[1].prediccion())
print(capas[1].precision(labels))
"""
"""
data = np.load("./mnist_dataset/mnist_custom_ds.npz")
images = data["images"]
labels = data["labels"]
print(images.shape)
input = images.reshape(10, -1)
input = input/255
"""

    


