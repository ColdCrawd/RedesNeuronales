import json
import numpy as np


import CapaDensa
import utils

data = np.load("./mnist_dataset/mnist_custom_ds.npz")
images = data["images"]
labels = data["labels"]
print(images.shape)
input = images.reshape(10, -1)
input = input/255


    
capas = utils.init("./mnist_dataset/mnist_mlp_pretty.json")
input_2 = [input[2]]
capas[0].forward(input)
#print(capas[0].output)
capas[1].forward(capas[0].output)
print(capas[1].output)
for value in capas[1].output:
    print(np.max(value), np.argmax(value))
