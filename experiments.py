import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree   # <-- your Assignment 1 DecisionTree
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_average_time = 1   


# Function to create fake data (supports 4 cases)
def generate_data(N, M, input_type, output_type):
    if input_type == "real" and output_type == "real":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
        return X, y

    elif input_type == "real" and output_type == "discrete":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(0, 2, size=N), dtype="category")
        return X, y

    elif input_type == "discrete" and output_type == "discrete":
        X = pd.DataFrame({i: pd.Series(np.random.randint(0, 2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(0, 2, size=N), dtype="category")
        return X, y

    elif input_type == "discrete" and output_type == "real":
        X = pd.DataFrame({i: pd.Series(np.random.randint(0, 2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
        return X, y


# Measure average fit + predict time
def measure_time(X_train, y_train, X_test, y_test):
    tree = DecisionTree(criterion="information_gain")   # your code expects "information_gain"

    # Training
    start_time = time.time()
    tree.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Prediction
    start_time = time.time()
    _ = tree.predict(X_test)
    predict_time = time.time() - start_time

    return train_time, predict_time


# Plot & save results
def plot_time_complexity(N_values, M_values, save_dir="plots"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    scenarios = [
        ("discrete", "discrete"),
        ("discrete", "real"),
        ("real", "discrete"),
        ("real", "real")
    ]
    scenario_labels = ["Discrete-Discrete", "Discrete-Real", "Real-Discrete", "Real-Real"]

    # Vary N for fixed M
    fig, axs = plt.subplots(len(M_values), 2, figsize=(15, 20))
    fig.suptitle("Time vs Number of Samples (N) - Criterion: information_gain")

    for j, M in enumerate(M_values):
        train_times = {label: [] for label in scenario_labels}
        predict_times = {label: [] for label in scenario_labels}

        for N in N_values:
            for scenario, label in zip(scenarios, scenario_labels):
                X, y = generate_data(N, M, scenario[0], scenario[1])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                avg_train_time, avg_predict_time = 0, 0
                for _ in range(num_average_time):
                    t_train, t_pred = measure_time(X_train, y_train, X_test, y_test)
                    avg_train_time += t_train
                    avg_predict_time += t_pred

                avg_train_time /= num_average_time
                avg_predict_time /= num_average_time

                train_times[label].append(avg_train_time)
                predict_times[label].append(avg_predict_time)

        # Training time
        for label in scenario_labels:
            axs[j, 0].plot(N_values, train_times[label], label=label, marker='o')
        axs[j, 0].set_title(f'Training Time vs N (M = {M})')
        axs[j, 0].set_xlabel('N')
        axs[j, 0].set_ylabel('Time (s)')
        axs[j, 0].grid(True)
        axs[j, 0].legend()

        # Prediction time
        for label in scenario_labels:
            axs[j, 1].plot(N_values, predict_times[label], label=label, marker='o')
        axs[j, 1].set_title(f'Prediction Time vs N (M = {M})')
        axs[j, 1].set_xlabel('N')
        axs[j, 1].set_ylabel('Time (s)')
        axs[j, 1].grid(True)
        axs[j, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{save_dir}/time_vs_N.png")
    plt.show()

    # Vary M for fixed N
    fig, axs = plt.subplots(len(N_values), 2, figsize=(15, 20))
    fig.suptitle("Time vs Number of Features (M) - Criterion: information_gain")

    for i, N in enumerate(N_values):
        train_times = {label: [] for label in scenario_labels}
        predict_times = {label: [] for label in scenario_labels}

        for M in M_values:
            for scenario, label in zip(scenarios, scenario_labels):
                X, y = generate_data(N, M, scenario[0], scenario[1])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                avg_train_time, avg_predict_time = 0, 0
                for _ in range(num_average_time):
                    t_train, t_pred = measure_time(X_train, y_train, X_test, y_test)
                    avg_train_time += t_train
                    avg_predict_time += t_pred

                avg_train_time /= num_average_time
                avg_predict_time /= num_average_time

                train_times[label].append(avg_train_time)
                predict_times[label].append(avg_predict_time)

        # Training time
        for label in scenario_labels:
            axs[i, 0].plot(M_values, train_times[label], label=label, marker='o')
        axs[i, 0].set_title(f'Training Time vs M (N = {N})')
        axs[i, 0].set_xlabel('M')
        axs[i, 0].set_ylabel('Time (s)')
        axs[i, 0].grid(True)
        axs[i, 0].legend()

        # Prediction time
        for label in scenario_labels:
            axs[i, 1].plot(M_values, predict_times[label], label=label, marker='o')
        axs[i, 1].set_title(f'Prediction Time vs M (N = {N})')
        axs[i, 1].set_xlabel('M')
        axs[i, 1].set_ylabel('Time (s)')
        axs[i, 1].grid(True)
        axs[i, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{save_dir}/time_vs_M.png")
    plt.show()


# Example run
N_values = [10, 20, 30, 40, 50]     # samples
M_values = [10, 20, 30, 40, 50]     # features
plot_time_complexity(N_values, M_values)
