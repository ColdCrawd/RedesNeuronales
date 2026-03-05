
class RedNeuronal:
    def __init__(self, capas):
        self.capas = capas
    
    def forward(self, entradas):
        for capa in self.capas:
            capa.forward(entradas)
            entradas = capa.output

    def prediccion(self):
        return self.capas[-1].prediccion()

    def precision(self, labels):
        return self.capas[-1].precision(labels)
    