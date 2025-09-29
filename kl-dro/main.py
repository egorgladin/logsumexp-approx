import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pickle
import os
import multiprocessing as mp
from torch.utils.data import DataLoader

from utils import load_and_preprocess_data, LinearRegressionModel, train_linreg, plot_convergence_bar


def batch_logsumexp(preds: torch.Tensor, targets: torch.Tensor):
    errors = (preds - targets) ** 2
    L = errors.squeeze(1)
    with torch.no_grad():
        p = torch.softmax(L, dim=0)
    loss = torch.sum(p * L)
    return loss


def softplus_approx(preds: torch.Tensor, targets: torch.Tensor, model: LinearRegressionModel, rho: float):
    errors = (preds - targets) ** 2
    L = errors.squeeze(1)
    loss = (1.0 / rho) * F.softplus(L - model.alpha + math.log(rho)).mean() + model.alpha
    return loss


def logsumexp_over_dataset(model: nn.Module, X, y):
    with torch.no_grad():
        preds = model(X)
        errors = (preds - y) ** 2
        L = errors.squeeze(1)
    return torch.logsumexp(L, dim=0).item() - math.log(len(y))


def train(dataset, batch_sz, lr, rho, seed):
    print(f"Running with batch_sz={batch_sz}, lr={lr}, rho={rho}, seed={seed}")
    torch.manual_seed(seed)
    train_loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)
    X, y = dataset.tensors

    model = LinearRegressionModel(input_dim=X.shape[1], with_alpha=rho is not None)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    logsumexp_vals = [logsumexp_over_dataset(model, X, y)]
    epochs_passed = [0]
    n_epochs = 50

    for epoch in range(1, n_epochs + 1):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = batch_logsumexp(preds, y_batch) if rho is None \
                else softplus_approx(preds, y_batch, model, rho)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            epochs_passed.append(epoch)
            logsumexp_vals.append(logsumexp_over_dataset(model, X, y))

    fname = 'batch_logsumexp' if rho is None else 'softplus_approx'
    fname += f'_batch{batch_sz}_lr{lr}_rho{rho}_seed{seed}.pickle'
    with open("trajectories/" + fname, "wb") as f:
        pickle.dump((epochs_passed, logsumexp_vals), f)
    print(f"Finished run with batch_sz={batch_sz}, lr={lr}, rho={rho}, seed={seed}")


def main():
    if not os.path.exists('trajectories'):
        os.makedirs('trajectories')

    dataset, X_np, y_np = load_and_preprocess_data()
    print("Loaded California housing dataset")

    train_linreg(X_np, y_np)
    print("Trained least squares linear regression for initialization")

    softplus_approx_params = [  # (batch_sz, lr, rho)
        (1000, 1e-5, 1e-3),
        (100,  1e-6, 1e-3),
        (10,   1e-7, 1e-3)
    ]
    batch_logsumexp_params = [
        (1000, 1e-5, None),  # rho==None corresponds to batch LogSumExp approach
        (100,  1e-5, None),
        (10,   1e-6, None)
    ]
    seeds = list(range(20))
    param_grid = softplus_approx_params + batch_logsumexp_params
    tasks = [(dataset, batch_sz, lr, rho, seed)
             for batch_sz, lr, rho in param_grid
             for seed in seeds]

    print("Running experiments...")
    with mp.Pool(processes=3) as pool:
        pool.starmap(train, tasks)
    print("Finished experiments.")

    for softplus_appr_param, batch_lse_param in zip(softplus_approx_params, batch_logsumexp_params):
        batch_sz = softplus_appr_param[0]
        plot_convergence_bar(softplus_appr_param, batch_lse_param, seeds, show_ylabel=batch_sz==10)
    print("Saved plots.")


if __name__ == "__main__":
    main()
