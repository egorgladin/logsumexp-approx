import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def load_and_preprocess_data():
    raw = fetch_california_housing()
    X = raw.data
    y = raw.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tX = torch.from_numpy(X_scaled.astype(np.float32))
    ty = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    dataset = TensorDataset(tX, ty)

    return dataset, X_scaled, y


def train_linreg(X, y, filename='linreg_weights.pt'):
    model = LinearRegression(fit_intercept=True).fit(X, y)
    state_dict = {
        'weights': torch.tensor(model.coef_, dtype=torch.float32),
        'bias': torch.tensor(model.intercept_, dtype=torch.float32)
    }
    torch.save(state_dict, filename)
    return state_dict


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim: int, with_alpha: bool = False, init_w: str = 'ridge'):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

        state_dict = torch.load('linreg_weights.pt')
        with torch.no_grad():
            self.linear.weight.copy_(state_dict['weights'].unsqueeze(0))
            self.linear.bias.copy_(state_dict['bias'])

        if with_alpha:
            self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def plot_convergence_bar(softplus_appr_param, batch_lse_param, seeds, show_ylabel=True):
    labels = ["Proposed approach", "Batch LogSumExp approach"]
    x_interest = np.array([20, 30, 40, 50])
    plt.figure(figsize=(8, 8))
    for lbl, (batch_sz, lr, rho) in zip(labels, [softplus_appr_param, batch_lse_param]):
        curves = []

        for seed in seeds:
            fname = 'batch_logsumexp' if rho is None else 'softplus_approx'
            fname += f'_batch{batch_sz}_lr{lr}_rho{rho}_seed{seed}.pickle'
            with open("trajectories/" + fname, "rb") as f:
                xs, ys = pickle.load(f)
            curves.append(ys)

        curves = np.array(curves)
        indices = [xs.index(x) for x in x_interest]
        mean_vals = curves[:, indices].mean(axis=0)
        std_vals = curves[:, indices].std(axis=0)
        offset = 0.1 if lbl[0]=='P' else -0.1
        plt.errorbar(x_interest + offset, mean_vals, yerr=std_vals, fmt='o-' if lbl[0]=='P' else 's-',
                     capsize=7, lw=2, label=lbl)

    plt.xlabel("Epochs", fontsize=15)
    plt.xticks(x_interest)
    plt.grid(True, which="both")

    ax = plt.gca()

    if show_ylabel:
        plt.ylabel("Objective value", fontsize=15)
    else:
        ax.set_yticklabels([])
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.ylim(0, 41)
    plt.legend(fontsize=15)
    plt.savefig(f"plots/batch{batch_lse_param[0]}.png", bbox_inches='tight', dpi=300)
    plt.close()
