import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import pickle
import math
import os
import numpy as np

from utils import prepare_dataset, plot_results, format_number
from models import SimpleCNN
from losses import inner_maximization, dro_loss, cross_entropy_loss
import multiprocessing as mp


def train(lam, rho, lr, seed, n_iters=int(1e5), n_checkpoints=10):
    torch.manual_seed(seed)
    random.seed(seed)

    X_train_cpu, y_train_cpu = torch.load("train_set_cpu.pt")

    model = SimpleCNN(with_alpha=rho is not None)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_sum_reduction = nn.CrossEntropyLoss(reduction='sum')
    criterion_no_reduction = None if lam is None else nn.CrossEntropyLoss(reduction='none')

    if lam is None:
        objective = 'crossentropy'
    else:
        objective = 'sumexp' if rho is None else 'approx'

    folder_name = f"trajectories/{objective}_weights_lr{lr:.0e}"
    if objective == 'approx':
        folder_name += f"_rho{rho}"
    folder_name += f"_seed{seed}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    torch.save(model.state_dict(), folder_name + f"/0.pth")

    loss_fn = cross_entropy_loss if objective == 'crossentropy' else dro_loss

    save_every = n_iters // n_checkpoints
    for step in range(1, n_iters + 1):
        i = random.randint(0, len(y_train_cpu) - 1)
        inputs, labels = X_train_cpu[i:i+1], y_train_cpu[i:i+1]
        loss = loss_fn(model, inputs, labels, criterion_sum_reduction, criterion_no_reduction,
                       lambda_=lam, rho=rho)
        if loss is None:
            print(f'    objective: {objective}, seed={seed}, iter {step}| Floating point exception occurred')
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == save_every // 2 or step % save_every == 0:
            print(f'    objective: {objective}, seed={seed} | iter {format_number(step)}/{format_number(n_iters)}')
            torch.save(model.state_dict(),
                       folder_name + f"/{step}.pth")


def eval_loss(model, cuda_loader, lam, cross_entropy_sum_reduction, cross_entropy_no_reduction):
    if lam is None:
        lam = 1.

    model.freeze_weights().eval()
    exponents = []
    for X, y in cuda_loader:
        U_star = inner_maximization(model, cross_entropy_sum_reduction, X, y, lam)
        preds = model(U_star)
        losses = cross_entropy_no_reduction(preds, y)
        costs = (X - U_star).pow(2).sum(dim=(1, 2, 3))
        exponents.append(losses - lam * costs)

    exponents = torch.cat(exponents)
    total_loss = lam * torch.logsumexp(exponents / lam, dim=0) - math.log(len(exponents))

    model.unfreeze_weights().train()
    return total_loss.item()


def eval_accuracy(model, loader):
    model.eval().to('cuda')
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def trajectory_to_metrics(lr, lam, rho, seeds, eval_batch_size=2000):
    X_train_cuda, y_train_cuda = torch.load("train_set_cuda.pt")
    X_val, y_val = torch.load("val_set.pt")
    X_test, y_test = torch.load("test_set.pt")
    train_loader_cuda = DataLoader(TensorDataset(X_train_cuda, y_train_cuda),
                                   batch_size=eval_batch_size,
                                   shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=eval_batch_size,
                            shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=eval_batch_size,
                             shuffle=False)

    criterion_sum_reduction = nn.CrossEntropyLoss(reduction='sum')
    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')

    if lam is None:
        objective = 'crossentropy'
    else:
        objective = 'sumexp' if rho is None else 'approx'

    model = SimpleCNN(with_alpha=rho is not None).cuda()

    folder_name_prefix = f"trajectories/{objective}_weights_lr{lr:.0e}"
    if rho:
        folder_name_prefix += f"_rho{rho}"

    loss_curves = []
    val_acc_curves = []
    test_acc_curves = []
    for seed in seeds:
        losses_fname = f'trajectories/{objective}_loss_lr{lr:.0e}_seed{seed}.pickle'
        val_acc_fname = f'trajectories/{objective}_val_acc_lr{lr:.0e}_seed{seed}.pickle'
        test_acc_fname = f'trajectories/{objective}_test_acc_lr{lr:.0e}_seed{seed}.pickle'
        if rho:
            losses_fname = f'trajectories/{objective}_loss_lr{lr:.0e}_rho{rho}_seed{seed}.pickle'
            val_acc_fname = f'trajectories/{objective}_val_acc_lr{lr:.0e}_rho{rho}_seed{seed}.pickle'
            test_acc_fname = f'trajectories/{objective}_test_acc_lr{lr:.0e}_rho{rho}_seed{seed}.pickle'

        if (os.path.exists(losses_fname) and os.path.exists(val_acc_fname)
                and os.path.exists(test_acc_fname)):
            with (open(losses_fname, "rb") as f1, open(val_acc_fname, "rb") as f2,
                  open(test_acc_fname, "rb") as f3):
                iter_numbers, losses = pickle.load(f1)
                iter_numbers, accs_val = pickle.load(f2)
                iter_numbers, accs_test = pickle.load(f3)
        else:
            losses, accs_val, accs_test = [], [], []
            iter_numbers = []

            folder_name = folder_name_prefix + f"_seed{seed}"
            weights_files = [(f, int(f.split('.')[0])) for f in os.listdir(folder_name)]
            weights_files.sort(key=lambda x: x[1])

            for f, i in tqdm(weights_files):
                state_dict = torch.load(os.path.join(folder_name, f))
                model.load_state_dict(state_dict)
                loss = None if objective == 'crossentropy'\
                    else eval_loss(model, train_loader_cuda, lam, criterion_sum_reduction, criterion_no_reduction)
                losses.append(loss)
                accs_val.append(eval_accuracy(model, val_loader))
                accs_test.append(eval_accuracy(model, test_loader))
                iter_numbers.append(i)

            with (open(losses_fname, "wb") as f1, open(val_acc_fname, "wb") as f2,
                  open(test_acc_fname, "wb") as f3):
                pickle.dump((iter_numbers, losses), f1)
                pickle.dump((iter_numbers, accs_val), f2)
                pickle.dump((iter_numbers, accs_test), f3)

        loss_curves.append(losses)
        val_acc_curves.append(accs_val)
        test_acc_curves.append(accs_test)

    loss_curves = np.array(loss_curves)
    loss_mean = None if objective == 'crossentropy' else loss_curves.mean(axis=0)
    loss_std = None if objective == 'crossentropy' else loss_curves.std(axis=0)

    val_curves = np.array(val_acc_curves)
    val_mean = val_curves.mean(axis=0)
    val_std = val_curves.std(axis=0)

    test_curves = np.array(test_acc_curves)
    test_mean = test_curves.mean(axis=0)
    test_std = test_curves.std(axis=0)

    res = {
        'loss': (iter_numbers, loss_mean, loss_std),
        'val_acc': (iter_numbers, val_mean, val_std),
        'test_acc': (iter_numbers, test_mean, test_std)
    }

    return objective, res


def main():
    if not os.path.exists('trajectories'):
        os.makedirs('trajectories')

    prepare_dataset()

    param_grid = [  # (lam, rho, lr)
        (1., 0.1, 1e-4),
        (1., None, 1e-5),  # rho==None corresponds to SumExp approach (if lam is not None)
        (None, None, 1e-3)  # lam==None corresponds to ERM
    ]
    seeds = list(range(10))
    tasks = [(lam, rho, lr, seed)
             for lam, rho, lr in param_grid
             for seed in seeds]

    print("Running experiments (in parallel)")
    with mp.Pool(processes=3) as pool:
        pool.starmap(train, tasks)
    print("Finished experiment runs.")

    print("Computing loss and accuracy for the generated trajectories")
    seeds_for_sumexp = [s for s in seeds if s != 8]
    results = dict()
    for lam, rho, lr in param_grid:
        is_sumexp = rho is None and lam is not None
        seeds_ = seeds_for_sumexp if is_sumexp else seeds
        objective, res = trajectory_to_metrics(lr, lam, rho, seeds_)
        results[objective] = res
    print("Loss and accuracy have been computed.")

    plot_results(results)
    print("Plots have been created.")


if __name__ == '__main__':
    main()
