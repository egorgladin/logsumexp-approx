## About

The paper ***Improved Stochastic Optimization of LogSumExp*** ([arxiv preprint](https://arxiv.org/abs/2509.24894)) introduces a LogSumExp approximation with tunable accuracy that can be efficiently optimized with stochastic methods.
This repository contains the experiments described in the paper, covering distributionally robust optimization (DRO) and entropy-regularized continuous optimal transport problems.
The experiments highlight numerical issues in existing LogSumExp optimization approaches and demonstrate how the proposed approximation addresses them, leading to superior performance.

To cite the preprint, use the following BibTeX entry:
```
@misc{gladin2025logsumexp,
  title={Improved Stochastic Optimization of LogSumExp},
  author={Egor Gladin and Alexey Kroshnin and Jia-Jie Zhu and Pavel Dvurechensky},
  year={2025},
  eprint={2509.24894},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2509.24894}
}
```

## Example Plots: Distributionally Robust Optimization (DRO) with Unbalanced OT

In one of the experiments, a small CNN is trained on MNIST with [noisy labels](https://github.com/gorkemalgan/corrupting_labels_with_distillation/).
DRO with unbalanced OT results in a LogSumExp-type objective, which we optimize using our proposed approach and a [baseline](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5d68a3f05ee2aae6a0fb2d94959082a0-Abstract-Conference.html).

On the figure below, the proposed approach (orange) minimizes the objective using a relatively large learning rate of $10^{-2}$.
The baseline (blue), however, requires a learning rate of $10^{-4}$ to avoid numerical overflow, as demonstrated by its rapid divergence (green dashed curve) for $\text{lr}=10^{-3}$. Consequently, the proposed method achieves lower objective values, while the baseline's small learning rate results in slow progress.

![loss curves example](/uot-dro/loss_curves_example.png)


## System Information

The following system was used to run all experiments in the paper:

- **OS**: Windows 10 Home, version 10.0.19045
- **CPU**: Intel Core i7-8550U, 4 cores / 8 threads
- **RAM**: 16 GB
- **GPU**: NVIDIA GeForce GTX 1050 with Max-Q Design
- **CUDA**: 12.1
- **Python**: 3.12.7
- **PyTorch CUDA**: available (torch.version.cuda = 12.1)

## Environment Setup

Before running the experiments, do the following:

```
# Create a virtual environment
python -m venv venv

# Activate it (on Linux, use `source venv/bin/activate`)
venv\Scripts\activate

# Upgrade pip (optional)
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Structure

- Executing `python plot_f_rho.py` produces Figure 1 from the paper, i.e., plot of $f_{\rho}(t)$ for different values of $\rho$. 
- Folders `neural_ot` and `eot` correspond to experiments in continuous entropy-regularized OT
with neural network parameterization and RKHS, while folders `kl-dro` and `uot-dro` correspond to experiments in distributionally robust optimization (DRO) with KL divergence and with unbalanced optimal transport, respectively. To run each
experiment, navigate to the dedicated folder and execute `python main.py`.
- Depending on your setup, experiments may take from a few minutes
up to **a few hours** due to repeated runs with different seeds.
