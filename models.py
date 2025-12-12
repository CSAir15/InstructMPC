import control
import numpy as np
import copy
import random

class predictor():
    def __init__(self, system_matrices, coefficient = (200, 150)):
        self.A, self.B, self.Q, self.R, self.T, self.horizon = system_matrices
        self.P, _, _ = control.dare(self.A, self.B, self.Q, self.R)
        self.D = np.matmul(np.linalg.inv(self.R + np.matmul(np.matmul(np.transpose(self.B), self.P), self.B)), np.transpose(self.B))
        self.H = np.matmul(self.B, self.D)
        self.F = self.A - np.matmul(self.H, np.matmul(self.P, self.A))
        self.F_list = [np.linalg.matrix_power(self.F, i) for i in range(self.T + 1)]

        self.real_w_trajectory = []
        self.pred_ws_trajectory = []
        self.contexts_trajectory = []
        self.horizon = 10
        self.a_1, self.a_2 = coefficient
        self.a_1_list, self.a_2_list = [], []

    @property
    def coefficient(self):
        return (self.a_1, self.a_2)

    def learning_rate(self, t, c = 0.05):
        return c / (np.sqrt(2 * (2 * self.horizon - 1) * (t + 1)))
    
    def save(self, t, horizon, real_w, pred_ws, context, loss_type = 'special'):
        self.a_1_list.append(self.a_1)
        self.a_2_list.append(self.a_2)
        if len(self.real_w_trajectory) == self.horizon:
            w_hat = self.pred_ws_trajectory.pop(0)
            self.pred_ws_trajectory.append(pred_ws)
            context_train = self.contexts_trajectory.pop(0)
            self.contexts_trajectory.append(context)
            self.update(t, self.real_w_trajectory, w_hat, context_train, horizon, loss_type)
            self.real_w_trajectory.pop(0)
            self.real_w_trajectory.append(real_w)
            

        else:
           self.real_w_trajectory.append(real_w)
           self.pred_ws_trajectory.append(pred_ws)
           self.contexts_trajectory.append(context)

    def compute_gradient_special_loss(self, ws_real, ws_pred, context, horizon):
        x_context, y_context = context
        phi = np.zeros(4)
        k_x, k_y = np.zeros((4, len(x_context))), np.zeros((4, len(y_context)))
        k_x[2], k_y[3] = x_context, y_context
        phi_grad_k_x = np.zeros(4)
        phi_grad_k_y = np.zeros(4)
        for tau in range(horizon):
            phi += np.matmul(np.transpose(self.F_list[tau]), np.matmul(self.P, ws_real[tau] - ws_pred[tau]))
            phi_grad_k_x += np.matmul(np.transpose(self.F_list[tau]), np.matmul(self.P, k_x[:, tau]))
            phi_grad_k_y += np.matmul(np.transpose(self.F_list[tau]), np.matmul(self.P, k_y[:, tau]))
        k_bar = np.column_stack((phi_grad_k_x, phi_grad_k_y))
        # gradient = - 2 * np.matmul(np.transpose(k_bar), np.matmul(self.H, phi))
        gradient = - 2 * np.matmul(np.transpose(k_bar), np.matmul((self.H + np.transpose(self.H)), phi))
        return gradient

    def compute_gradient_mse_loss(self, ws_real, ws_pred, context, horizon):
        x_context, y_context = context
        k_x, k_y = np.zeros((4, len(x_context))), np.zeros((4, len(y_context)))
        k_x[2], k_y[3] = x_context, y_context
        gradient = np.zeros(2)
        for tau in range(horizon):
            k_tau = np.column_stack((k_x[:, tau], k_y[:, tau]))
            gradient -= 2 * np.matmul(np.transpose(k_tau), (ws_real[tau] - ws_pred[tau]))
        # gradient /= self.learning_rate(t)
        if horizon != 0:
            gradient /= horizon
        gradient *= 0.02
        return gradient

    def compute_gradient_mae_loss(self, ws_real, ws_pred, context, horizon):
        x_context, y_context = context
        k_x, k_y = np.zeros((4, len(x_context))), np.zeros((4, len(y_context)))
        k_x[2], k_y[3] = x_context, y_context
        
        gradient = np.zeros(2)
        
        for tau in range(horizon):
            k_tau = np.column_stack((k_x[:, tau], k_y[:, tau]))
            error_vector = ws_real[tau] - ws_pred[tau]
            mae_term = np.matmul(np.transpose(k_tau), np.sign(error_vector))
            
            gradient -= mae_term
        if horizon != 0:
            gradient /= horizon
        gradient *= 120
        return gradient
          
    def update(self, t, ws_real, ws_pred, context, horizon, loss_type = 'special'):
        if loss_type == 'special':
            gradient = self.compute_gradient_special_loss(ws_real, ws_pred, context, horizon)
        elif loss_type == 'mse':
            gradient = self.compute_gradient_mse_loss(ws_real, ws_pred, context, horizon)
        else:
            gradient = self.compute_gradient_mae_loss(ws_real, ws_pred, context, horizon)
        learning_rate = self.learning_rate(t)
        self.a_1 -= learning_rate * gradient[0]
        self.a_2 -= learning_rate * gradient[1]



class IMPC_tracking():
    def __init__(self, trajectory, system_matrices, coefficient = (200, 150)):
        self.trajectory = trajectory #a function
        self.A, self.B, self.Q, self.R, self.T, self.horizon = system_matrices
        self.P, _, _ = control.dare(self.A, self.B, self.Q, self.R)
        self.D = np.matmul(np.linalg.inv(self.R + np.matmul(np.matmul(np.transpose(self.B), self.P), self.B)), np.transpose(self.B))
        self.H = np.matmul(self.B, self.D)
        self.F = self.A - np.matmul(self.H, np.matmul(self.P, self.A))
        self.F_list = [np.linalg.matrix_power(self.F, i) for i in range(self.T + 1)]


        self.w_without_perterbation = self.w_without_disturbances()

        self.context = self.generate_context()

        self.real_w = self.create_disturbances()
        # self.pred_w = self.w_without_perterbation
        self.cost = np.zeros(self.T)
        self.predictor = predictor(system_matrices, coefficient)



    def w_without_disturbances(self):
        w = np.zeros((self.T, np.shape(self.A)[1]))
        for t in range(self.T):
            y_1, y_2 = self.trajectory(t)
            y_3, y_4 = self.trajectory(t + 1)

            w[t] = np.matmul(self.A, np.array([y_1, y_2, 0, 0])) - np.array([y_3, y_4, 0, 0])
        
        return w

    def generate_context(self, var = (20, 20)):
        x_context = np.random.uniform(-var[0], var[0], self.T)
        y_context = np.random.uniform(-var[1], var[1], self.T)
        return (x_context, y_context)

    def create_disturbances(self):
        x_context, y_context = self.context[0], self.context[1]
        real_w = copy.deepcopy(self.w_without_perterbation)

        for t in range(self.T):
            real_w[t][2] = real_w[t][2] - 0.2 * x_context[t] + random.gauss(0, 1)
            real_w[t][3] = real_w[t][3] - 0.2 * y_context[t] + random.gauss(0, 1)
        return real_w

    def predict_disturbances(self, t, horizon, run_baseline = False):
        x_context, y_context = self.context[0][t : t + horizon], self.context[1][t : t + horizon]
        pred_w = copy.deepcopy(self.w_without_perterbation)[t : t + horizon]
        coefficient = self.predictor.coefficient
        if run_baseline:
            coefficient = [-0.2, -0.2]

        for i in range(len(pred_w)):
            pred_w[i][2] = pred_w[i][2] + coefficient[0] * x_context[i]
            pred_w[i][3] = pred_w[i][3] + coefficient[1] * y_context[i]

        return pred_w, (x_context, y_context)

    def mpc_solver(self, x, t, w_t, w_hat):
        k = min(self.horizon, self.T - 1 - t) 
        G = np.zeros(4)
        E = np.matmul(self.P, np.matmul(self.A, x))

        for tau in range(k):
            G += np.matmul(np.transpose(self.F_list[tau]), np.matmul(self.P, w_hat[tau])) 
        u_t = -np.matmul(self.D, E) - np.matmul(self.D, G)
        if not t == 0:
            self.cost[t] = self.cost[t - 1] + x.T @ self.Q @ x + u_t.T @ self.R @ u_t
        else:
            self.cost[0] = x.T @ self.Q @ x + u_t.T @ self.R @ u_t
        x_next = np.matmul(self.A, x) + np.matmul(self.B, u_t) + w_t
    
        return x_next
    
    def run_mpc(self, loss_type = 'special', run_baseline = False):
        state_x = np.zeros((self.T, np.shape(self.A)[1]))
        robot_path = np.zeros((self.T, 2))

        for t in range(self.T):
            k = min(self.horizon, self.T - 1 - t)
            w_t = self.real_w[t]
            # w_hat = self.pred_w[t : t + k]
            if not run_baseline:
                w_hat, context = self.predict_disturbances(t, k)
                self.predictor.save(t, k, w_t, w_hat, context, loss_type)
            else:
                w_hat, context = self.predict_disturbances(t, k, run_baseline)
            

            if t < self.T - 1:
                state_x[t + 1] = self.mpc_solver(x = state_x[t], t = t, w_t = w_t, w_hat = w_hat)
        
        for t in range(self.T):
            robot_path[t] = state_x[t][:2] + self.trajectory(t)

        return robot_path