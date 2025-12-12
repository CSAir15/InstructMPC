from models import *
from plot import *
from drone_inspection import *
from _DEFAULT_PARAS import *
import numpy as np
import argparse
from matplotlib import pyplot as plt

def get_configs():

    parser = argparse.ArgumentParser(
        description='InstructMPC configurations')

    parser.add_argument('--mode', default='drone_tracking', type=str,
                        help='drone_tracking or SoC_tracking')
    parser.add_argument('--exp_type', default='demo', type=str,
                        help='Type of the experiment. demo or grid_inspection; only for mode = "drone_tracking" ')
    parser.add_argument('--T', default=1260,
                        type=int, help='Number of time slots (Only for mode = "drone_tracking" type = "demo")')
    parser.add_argument('--horizon', default=10,
                        type=int, help='MPC horizon')
    parser.add_argument('--M', default=5,
                        type=int, help='Number of Monte Carlo tests')
    configs = parser.parse_args()

    return configs

def main():
    configs = get_configs()
    print(configs)
    mode = configs.mode
    exp_type = 'grid_inspection'
    horizon = configs.horizon
    M = configs.M

    A, B, Q, R, T, predictor_coefficients, coordinates  = return_paras(mode, exp_type)
    if mode == 'drone_tracking' and exp_type == 'demo':
        T = configs.T
    elif mode == 'drone_tracking' and exp_type == 'grid_inspection':
        T = 500
    P, _, _ = control.dare(A, B, Q, R)
    D = np.matmul(np.linalg.inv(R + np.matmul(np.matmul(np.transpose(B), P), B)), np.transpose(B))
    H = np.matmul(B, D)
    F = A - np.matmul(H, np.matmul(P, A))
    F_list = [np.linalg.matrix_power(F, i) for i in range(T + 1)]

    system_matrices = (A, B, Q, R, T, horizon)

    if mode == 'drone_tracking' and exp_type == 'demo':
        coords = [coordinates(t) for t in range(T)]
        impc = IMPC_tracking(coordinates, system_matrices, coefficient = predictor_coefficients)
        k = impc.run_mpc()
        y1_vals, y2_vals = zip(*coords)
        plt.figure(figsize=(10, 10))
        plt.plot(k[:, 0], k[:, 1], '-', label='Instruct MPC with Fine Tuning', color = 'blue')
        plt.plot(y1_vals, y2_vals, 'k--', label='Unknown Trajectory')
        plt.legend()
        plt.show()

    elif mode == 'drone_tracking' and exp_type == 'grid_inspection':
        
        drone_inspection(M, system_matrices, predictor_coefficients)


main()