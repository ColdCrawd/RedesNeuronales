import numpy as np
import Activacion as FA

class CapaDensa:
    def __init__(self, n_neuronas: int, n_entradas: int, activacion):
        self.pesos = np.random.randn(n_neuronas, n_entradas) * 0.01
        self.sesgos = np.zeros((1, n_neuronas))
        self.activacion = activacion

    def __init__(self, pesos, sesgos, activacion):
        self.pesos = pesos
        self.sesgos = sesgos
        self.activacion = activacion

    def forward(self, entradas):
        transferencia = np.dot(entradas, self.pesos) + self.sesgos
        self.output = self.activacion(transferencia)

    def prediccion(self):
        return np.argmax(self.output, axis=1)

    def precision(self, labels):
        predicho = np.argmax(self.output, axis=1)
        return np.mean(predicho == labels)
    