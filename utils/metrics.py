import numpy as np

def ensure_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    elif hasattr(array, 'detach'):
        return array.detach().cpu().numpy()
    else:
        raise TypeError("Input type not supported. Expected numpy.ndarray or PyTorch Tensor.")

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    pred = ensure_numpy(pred)
    true = ensure_numpy(true)
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    pred = ensure_numpy(pred)
    true = ensure_numpy(true)
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true, epsilon=1e-2):
    pred = ensure_numpy(pred)
    true = ensure_numpy(true)
    mask = true > epsilon
    if not np.any(mask):
        return 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe
