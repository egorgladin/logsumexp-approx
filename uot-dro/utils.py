import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import urllib.request
from matplotlib.ticker import MaxNLocator


def format_number(x):
    if x >= 1e6:
        return f'{x * 1e-6:.0f}M'
    elif x >= 1e3:
        return f'{x * 1e-3:.0f}k'
    else:
        return f'{x:.0f}'


def prepare_dataset(val_test_device='cuda'):
    train_path_cpu, train_path_cuda = "train_set_cpu.pt", "train_set_cuda.pt"
    val_path, test_path = "val_set.pt", "test_set.pt"

    if not (os.path.exists(train_path_cpu) and os.path.exists(train_path_cuda)
            and os.path.exists(val_path) and os.path.exists(test_path)):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load full training set (60k samples)
        print("Donwloading MNIST (train and val)")
        full_train_dataset = torchvision.datasets.MNIST(
            root='.',
            train=True,
            download=True,
            transform=transform
        )

        filename = "feature-dependent_25_ytrain.npy"
        if not os.path.exists(filename):
            print("Donwloading noisy labels...", end=' ')
            url = "https://github.com/gorkemalgan/corrupting_labels_with_distillation/raw/refs/heads/master/noisylabels/mnist/feature-dependent_25_ytrain.npy"
            urllib.request.urlretrieve(url, filename)
            print('Done.')

        y_train_noisy = np.load(filename)
        full_train_dataset.targets = y_train_noisy.tolist()

        # Split into train (50k) and validation (10k)
        train_size = 50000
        val_size = 10000
        torch.manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset,
            [train_size, val_size]
        )

        # Load test set (10k samples)
        test_dataset = torchvision.datasets.MNIST(
            root='.',
            train=False,
            download=True,
            transform=transform
        )

        # Convert and save training set (50k)
        X_train = torch.stack([img for img, _ in train_dataset])
        y_train = torch.tensor([full_train_dataset.targets[i] for i in train_dataset.indices])
        torch.save((X_train, y_train), train_path_cpu)
        torch.save((X_train.to('cuda'), y_train.to('cuda')), train_path_cuda)

        # Convert and save validation set (10k)
        X_val = torch.stack([img for img, _ in val_dataset]).to(val_test_device)
        y_val = torch.tensor([full_train_dataset.targets[i] for i in val_dataset.indices]).to(val_test_device)
        torch.save((X_val, y_val), val_path)

        # Convert and save test set (10k)
        X_test = torch.stack([img for img, _ in test_dataset]).to(val_test_device)
        y_test = torch.tensor([label for _, label in test_dataset]).to(val_test_device)
        torch.save((X_test, y_test), test_path)
        print("Dataset has been prepared.")


def plot_results(results):
    objective_to_lbl = {
        'crossentropy': 'ERM',
        'sumexp': 'SumExp Minimization',
        'approx': 'Proposed Approach'
    }

    # Plot loss
    plt.figure(figsize=(8, 8))
    for objective, res in results.items():
        if objective != 'crossentropy':
            iter_numbers, loss_mean, loss_std = res['loss']
            plt.plot(iter_numbers, loss_mean, label=objective_to_lbl[objective])
            plt.fill_between(
                iter_numbers,
                loss_mean - loss_std,
                loss_mean + loss_std,
                alpha=0.15
            )

    tirck_formatter = FuncFormatter(lambda x, pos: format_number(x))
    plt.gca().xaxis.set_major_formatter(tirck_formatter)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Train Loss', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig('plots/loss.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Plot accuracy
    plot_accuracy(results, objective_to_lbl, 'plots/accuracy_erm.png', exclude=['sumexp', 'approx'])
    plot_accuracy(results, objective_to_lbl, 'plots/accuracy.png', exclude=['crossentropy'])


def plot_accuracy(results, objective_to_lbl, fname, exclude=()):
    plt.figure(figsize=(8, 8))
    for objective, res in results.items():
        if objective not in exclude:
            iter_numbers, val_mean, val_std = res['val_acc']
            iter_numbers, test_mean, test_std = res['test_acc']
            plt.plot(iter_numbers, val_mean, label=objective_to_lbl[objective] + ', Val Accuracy')
            plt.fill_between(
                iter_numbers,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.15
            )
            plt.plot(iter_numbers, test_mean, label=objective_to_lbl[objective] + ', Test Accuracy', linestyle='--')
            plt.fill_between(
                iter_numbers,
                test_mean - test_std,
                test_mean + test_std,
                alpha=0.15
            )

    plt.ylim(70.5, 89.5)
    tirck_formatter = FuncFormatter(lambda x, pos: format_number(x))
    plt.gca().xaxis.set_major_formatter(tirck_formatter)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.xlabel('Iteration', fontsize=15)

    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if 'crossentropy' in exclude:
        ax.set_yticklabels([])
    else:
        plt.ylabel("Accuracy", fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()
