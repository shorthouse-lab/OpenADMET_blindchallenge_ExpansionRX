# util.py
from sklearn.metrics import mean_squared_error, log_loss
import numpy as np

def evaluate(target, prediction, task_type):
    """Return scalar loss for Optuna objective.
    Ensures finite value; returns large penalty if invalid.
    """
    try:
        if task_type == 'regression':
            val = mean_squared_error(target, prediction)
        else:
            # Clip prediction probabilities slightly away from 0/1 to avoid log_loss inf.
            prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
            val = log_loss(target, prediction)
        if not np.isfinite(val):
            return 1e12
        return float(val)
    except Exception as exc:
        print(f"[evaluate] Failed to compute loss ({exc}); assigning penalty.")
        return 1e12

def get_task_type(Y):
    """Infer task type.
    Classification only if all labels are in {0,1} and <=2 unique values.
    Otherwise regression.
    """
    unique = set(np.asarray(Y).tolist())
    if len(unique) <= 2 and unique.issubset({0, 1}):
        return 'classification'
    return 'regression'
