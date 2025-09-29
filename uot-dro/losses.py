import torch
import torch.nn.functional as F
import math


def inner_maximization(model, criterion, X, y, lam, lr=None, momentum=0.4, n_steps=5):
    if lam is None:
        lam = 1.
    if lr is None:
        lr = 0.1 / lam
    U = X.clone().requires_grad_(True)
    v = torch.zeros_like(U)

    for step in range(n_steps):
        U_ahead = U + momentum * v
        preds = model(U_ahead)
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            return None

        loss = criterion(preds, y) - lam * (X - U).pow(2).sum()
        grad, = torch.autograd.grad(loss, U, create_graph=False)
        v = momentum * v + lr * grad
        with torch.no_grad():
            U += v

    return U.detach()


def dro_loss(model, X, y, criterion_sum_reduction, criterion_no_reduction, lambda_=1.0, beta=1.0, rho=0.1):
    model.freeze_weights()
    U_star = inner_maximization(model, criterion_sum_reduction, X, y, lambda_)
    model.unfreeze_weights()
    if U_star is None:
        return None

    preds = model(U_star)
    losses = criterion_no_reduction(preds, y)
    costs = (X - U_star).pow(2).sum(dim=(1, 2, 3))
    if rho is None:
        total = torch.exp((losses - lambda_ * costs) / (lambda_ * beta)).mean()
    else:
        exponent = (losses - lambda_ * costs - model.alpha) / (lambda_ * beta) + math.log(rho)
        total = (lambda_ * beta / rho) * F.softplus(exponent).mean() + model.alpha
    return total


def cross_entropy_loss(model, X, y, criterion_sum_reduction, criterion_no_reduction=None, lambda_=None, rho=None, beta=None):
    preds = model(X)
    loss = criterion_sum_reduction(preds, y)
    return loss
