import json
import numpy as np


from CapaDensa import CapaDensa

import sys
import matplotlib.pyplot as plt

from RedNeuronal import RedNeuronal
from utils import cargar_capas, cargar_entrada
import random

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
        incorrectas.append((imagen, label, red.prediccion()[0]))

print( presiciones /len(labels))



import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))


min_random_images = min(len(incorrectas), 16)
imagenes_rnd = random.sample(incorrectas, min_random_images)

for i in range(min_random_images):
    plt.subplot(4, 4, i+1)
    img, lbl, prediccion = imagenes_rnd[i]
    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.title(f"Label: {lbl} | P: {prediccion}")
    plt.axis("off")
plt.show()
