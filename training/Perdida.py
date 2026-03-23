import numpy as np

#Hay varias funciones de perdida
#MSE, MAE, HubberLoss, EntropiaCruzadaCategorica, EntropiaCruzadaBinaria

#MSE
def mse(y_true, y_pred):
    return np.mean((y_true-y_pred) ** 2)

#esto suponiendo que son batches
def mse_nnp(y_true , y_pred):
    return 1/len(y_true) * sum([ sum([(t - p) ** 2 for t, p in zip(true, pred)])/len(true) for true, pred in zip(y_true, y_pred)])

y_true = np.array([[1.0, 0.0, 0.0],
	  [0.0, 1.0, 0.0]])
y_pred = np.array([[0.8, 0.1, 0.1], 
	  [0.1, 0.7, 0.2]])
print(f"MSE total {mse(y_true, y_pred):.2f}")
print(f"MSE_NNP total {mse_nnp(y_true, y_pred):.2f}")


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
