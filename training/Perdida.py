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

def mae_nnp(y_true, y_pred):
    return sum([ sum([abs(t - p) for t, p in zip(true, pred)])/len(true) for true, pred in zip(y_true, y_pred)]) / len(y_true)

print(f"MAE total {mae(y_true, y_pred):.2f}")
print(f"MAE_NNP total {mae_nnp(y_true, y_pred):.2f}")


def hubber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    decide = abs(error) <= delta
    return np.mean(np.where(decide, 0.5*error**2, delta* (abs(error) - 0.5 * delta)))


def hubber_loss_nnp(y_true, y_pred, delta=1.0):	
    return sum( [ sum([ 0.5 * (t-p) ** 2 if abs(t-p) <= delta else delta * ((abs(t-p)) - 0.5 * delta) for t, p in zip(true, pred)]) / len(true)  for true, pred in zip(y_true, y_pred)]) / len (y_true)

print(f"MAE total {hubber_loss(y_true, y_pred):.2f}")
print(f"MAE_NNP total {hubber_loss_nnp(y_true, y_pred):.2f}")

import math

#Entropia cruzada categorica
def cce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis = 1))

def cce_nnp(y_true, y_pred):
    return sum([sum([t * math.log(max(min(p, 1-1e-7), 1e-7)) for t, p in zip(true, pred)]) for true, pred in zip(y_true, y_pred)])/len(y_true)

print(f"MAE total {cce(y_true, y_pred):.2f}")
print(f"MAE_NNP total {cce_nnp(y_true, y_pred):.2f}")
 
