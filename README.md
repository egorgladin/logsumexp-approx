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
- Folders `eot`, `kl-dro` and `uot-dro` correspond to experiments in continuous entropy-regularized optimal transport, distributionally
robust optimization (DRO) with KL divergence, and DRO with unbalanced optimal transport, respectively. To run each
experiment, navigate to the dedicated folder and execute `python main.py`.
- Depending on your setup, experiments may take from a few minutes
up to **a few hours** due to repeated runs with different seeds.
