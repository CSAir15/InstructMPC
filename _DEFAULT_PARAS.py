import numpy as np
import math

def tracking_coordinates(t):

    y_1 = (t / 38.2) * math.sin(t/38.2)
    y_2 = (t / 38.2) * (math.cos(t/38.2) ** 2)

    return y_1, y_2

def return_paras(mode, exp_type = None):
    if mode == 'drone_tracking' and exp_type == 'demo':
        A = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
        Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        R = np.array([[1e-4, 0], [0, 1e-4]])
        T = 1260
        predictor_coefficients = (2, 2)
        coordinates = tracking_coordinates
        return A, B, Q, R, T, predictor_coefficients, coordinates
    
    elif mode == 'drone_tracking' and exp_type == 'grid_inspection':
        A = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
        Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        R = np.array([[1e-4, 0], [0, 1e-4]])
        T = None
        predictor_coefficients = (200, 150)
        coordinates = None
        return A, B, Q, R, T, predictor_coefficients, coordinates
    else:
        A, B, Q, R, T, coordinates = None, None, None, None, None, None
        return A, B, Q, R, T, coordinates