import numpy as np

class CapaDensa:
    def __init__(self, n_neuronas: int, n_entradas: int, activacion: function):
        self.pesos = np.random.randn(n_neuronas, n_entradas) * 0.01
        self.sesgos = np.zeros((1, n_neuronas))
        self.activacion = activacion

    def forward(self, entradas):
        resultado = np.dot(entradas, self.pesos) + self.sesgos
        self.output = self.activation(resultado)
