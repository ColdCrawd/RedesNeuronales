import numpy as np
import json
from CapaDensa import CapaDensa

import Activacion as FA

MAPA_ACTIVACIONES = {
    "relu": FA.relu,
    "sigmoid": FA.sigmoid,
    "tanh": FA.tanh,
    "step": FA.step_function,
    "leaky_relu": FA.leaky_relu,
    "telu": FA.telu,
    "softmax": FA.softmax
}

def cargar_capas(json_path):
    with open(json_path, "r") as data:
        red = json.load(data)
    if "layers" not in red.keys():
        raise ValueError("No hay capas, tonoto")
    
    capas = []
    for layer in red['layers']:
        if "units" not in layer.keys():
            raise ValueError("No hay capas, tonoto")
        if "activation" not in layer.keys():
            raise ValueError("No hay activacion, tonoto")
        if "W" not in layer.keys():
            raise ValueError("No hay pesos, tonoto")
        if "b" not in layer.keys():
            raise ValueError("No hay Sesgos, tonoto")
        capas.append(CapaDensa(layer["W"], layer["b"],  MAPA_ACTIVACIONES[layer["activation"]]))
    return capas

def cargar_entrada(file_path):
    data = np.load(file_path)
    if "images" not in data.keys() or "labels" not in data.keys():
        raise ValueError("ERRORRROROROROR")
    imagenes = data["images"]
    labels = data["labels"]
    
    if len(imagenes.shape) == 3:
        input = imagenes.reshape(imagenes.shape[0], -1)
    elif len(imagenes.shape) == 2:
        input = imagenes.reshape(-1)
    else:
        raise ValueError("Valor no intente caso")

    return input/255, labels

