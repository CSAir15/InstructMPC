import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import re
from pathlib import Path

data = pd.read_csv("opencem_sample.csv", sep=";", header=0)
data["start_ts"] = pd.to_datetime(data["start_ts"])
data["end_ts"] = pd.to_datetime(data["end_ts"])

def mpc_1d_unconstrained(a, b, q, r, P_term, x0, w_hat):
    w_hat = np.asarray(w_hat, dtype=float).reshape(-1)
    N = w_hat.size
    P = float(P_term)
    s = 0.0
    for k in range(N-1, -1, -1):
        w = float(w_hat[k])
        S = r + (b*b) * P
        K = (b * P * a) / S
        kff = (b * (P * w + s)) / S
        P = q + (a*a) * P - ((a*b*P)**2) / S
        s = a * (s + P * w) - (a*b*P / S) * b * (s + P * w)
        if k == 0:
            u0 = -K * float(x0) - kff
            return u0
    return 0.0

def run_experiment(A = 1., B = 1., Q = 1.e-2, R = 1.e-4, eta = 1.e-3, intercept_guess = None,
                   features = ["group_computer_effort", "roof_server_effort", "group_computer_gpu_effort"],
                   start_time = "2025-11-24T19:34:00+08:00", T_hours = 96., k = 60, verbose = False):
    start_time = pd.to_datetime(start_time)
    experiment_data = data[(data["start_ts"] >= start_time) & (data["start_ts"] < start_time + pd.Timedelta(hours=T_hours))]
    T = len(experiment_data)
    P = float(scipy.linalg.solve_discrete_are([[A]], [[B]], [[Q]], [[R]])[0, 0])
    H = (B * B) / (R + (B * B) * P)
    F = A - H * P * A
    w = -experiment_data["power_demand"].to_numpy(dtype=np.float64)
    if intercept_guess is None:
        intercept_guess = np.average(w)
    theta = np.array([-1.] * len(features) + [intercept_guess], dtype=np.float64)
    if verbose:
        print(f"Running experiment with A={A}, B={B}, Q={Q}, R={R}, \\eta={eta}, \\theta_0={theta}, features={features}, start_time={start_time}, T_hours={T_hours}, k={k}, verbose={verbose}")
    d = experiment_data[features].to_numpy(dtype=np.float64)
    if features != ["power_demand"]:
        d = (d - d.mean(axis=0)) / d.std(axis=0)
    Phi_buf = []
    theta_buf = []
    x = 0.
    cost = 0.

    state_record = []
    action_record = []
    state_norm_record = []
    cumulative_cost_record = []

    for t in range(T):
        theta_buf.append(theta.copy())
        Tau = min(t + k, T)
        d_t = d[t:Tau]
        Phi = np.hstack([d_t, np.ones((Tau - t, 1))])
        Phi_buf.append(Phi)
        w_hat = Phi @ theta
        u0 = mpc_1d_unconstrained(A,B,Q,R,P,x,w_hat)
        cost = cost + x * Q * x + u0 * R * u0
        x = A * x + B * u0 + w[t]
        s = t - k + 1
        if s >= 0:
            Phi_s = Phi_buf[s]
            theta_s = theta_buf[s]
            N_s = Phi_s.shape[0]
            w_true_seg = w[s:s+N_s]
            w_hat_seg  = Phi_s @ theta_s
            alpha = P * (F ** np.arange(N_s, dtype=np.float64))
            psi_hat = alpha @ (w_true_seg - w_hat_seg)
            v = (alpha[:, None] * Phi_s).sum(axis=0)
            grad = -2.0 * H * psi_hat * v
            theta = theta - eta * grad
        if verbose:
            print(x, u0, experiment_data["power_demand"][t], theta)
        action_record.append(u0)
        state_record.append(x)
        state_norm_record.append(abs(x))
        cumulative_cost_record.append(cost)
    cost = cost + x * P * x
    cumulative_cost_record.append(cost)
    if verbose:
        print(f"cost={cost}, \\theta_T={theta}\n")
    return state_record, state_norm_record, action_record, cumulative_cost_record, theta

def sanitize(s):
    """Make a safe filename stem."""
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', s)

def plot_experiments(results, out="./output2"):
    save_dir = Path(out)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_state, ax_state = plt.subplots(figsize=(8, 3))
    fig_action, ax_action = plt.subplots(figsize=(8, 3))
    fig_cumcost, ax_cum = plt.subplots(figsize=(8, 3))
    fig_cumcostlog, ax_cumlog = plt.subplots(figsize=(8, 3))
    i = 0
    T = 750
    for exp_name, data in results.items():
        if i < 4:
            ax_state.plot(range(T), data["states"][0:T], label=exp_name)

        ax_action.plot(range(T), data["actions"][0:T], label=exp_name)
        #if i <= 4:
        ax_cum.plot(range(T), data["costs"][0:T], label=exp_name)
        ax_cumlog.plot(range(T), data["costs"][0:T], label=exp_name)

        i = i + 1

    title = "Battery Management with Target SOC"
    ax_state.set_xlabel("Time t")
    ax_state.set_ylabel("State Norm ‖xₜ‖")
    ax_state.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_state.set_ylim((0.0, 200.0))
    ax_state.set_title(f"{title} — State Norms")
    ax_state.grid(True, linestyle="--", alpha=0.4)
    ax_state.legend()

    ax_action.set_xlabel("Time t")
    ax_action.set_ylabel("Action Norm ‖uₜ‖")
    ax_action.set_title(f"{title} — Action Norms")
    ax_action.grid(True, linestyle="--", alpha=0.4)
    ax_action.legend()

    ax_cum.set_xlabel("Time t")
    ax_cum.set_ylabel("Cumulative Cost")
    ax_cum.set_title(f"{title} — Cumulative Cost")
    ax_cum.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_cum.set_ylim((0.0, 30000.0))
    ax_cum.grid(True, linestyle="--", alpha=0.4)
    ax_cum.legend()

    ax_cumlog.set_xlabel("Time t")
    ax_cumlog.set_ylabel("Cumulative Cost")
    ax_cumlog.set_title(f"{title} — Cumulative cost (log)")
    ax_cumlog.set_yscale("log")
    ax_cumlog.grid(True, linestyle="--", alpha=0.4)
    ax_cumlog.legend()

    fig_state.tight_layout()
    fig_action.tight_layout()
    fig_cumcost.tight_layout()
    fig_cumcostlog.tight_layout()

    f_state = save_dir / "multi_states.pdf"
    f_action = save_dir / "_multi_actions.pdf"
    f_cum = save_dir / "multi_cumulative_costs.pdf"
    f_log_cum = save_dir / "multi_cumulative_costs_log.pdf"

    fig_state.savefig(f_state, dpi=200)
    fig_action.savefig(f_action, dpi=200)
    fig_cumcost.savefig(f_cum, dpi=200)
    fig_cumcostlog.savefig(f_log_cum, dpi=200)

    plt.close(fig_state)
    plt.close(fig_action)
    plt.close(fig_cumcost)
    plt.close(fig_cumcostlog)

    return {
        "states": f_state,
        "actions": f_action,
        "cumulative_costs": f_cum,
    }

eta = 1e-1
results = {}
states, state_norms, actions, cum_cost, theta = run_experiment(eta=eta, features=["power_demand"], intercept_guess=0)
results["Perfect Prediction"] = {"actions" : actions, "costs": cum_cost, "states": state_norms}
states, state_norms, actions, cum_cost, theta = run_experiment(eta=eta, features=["roof_server_files", "roof_server_cpus", "group_computer_files", "group_computer_cpus", "group_computer_parameters", "group_computer_optimizer"])
results["Classic Contextual MPC"] = {"actions" : actions, "costs": cum_cost, "states": state_norms}
states, state_norms, actions, cum_cost, theta = run_experiment(eta=eta, features=["roof_server_files", "roof_server_cpus", "roof_server_effort", "group_computer_files", "group_computer_cpus", "group_computer_effort", "group_computer_parameters", "group_computer_optimizer", "group_computer_gpu_effort"])
#results["InstructMPC + Classic Metadata"] =  {"actions" : actions, "costs": cum_cost, "states": state_norms}
states, state_norms, actions, cum_cost, theta = run_experiment(eta=eta)
results["InstructMPC"] =  {"actions" : actions, "costs": cum_cost, "states": state_norms}
states, state_norms, actions, cum_cost, theta = run_experiment(eta=eta, features=[])
results["MPC without context"] =  {"actions" : actions, "costs": cum_cost, "states": state_norms}
states, state_norms, actions, cum_cost, theta = run_experiment(eta=0, features=[])
results["Fixed average prediction"] =  {"actions" : actions, "costs": cum_cost, "states": state_norms}
states, state_norms, actions, cum_cost, theta = run_experiment(eta=0, features=[], intercept_guess=0)
results["Fixed zero prediction"] =  {"actions" : actions, "costs": cum_cost, "states": state_norms}

plot_experiments(results)
