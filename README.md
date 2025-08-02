# Sum-of-norms regularized Nonnegative Matrix Factorization (SONNMF)

[![arXiv](https://img.shields.io/badge/arXiv-2407.00706-b31b1b.svg)](https://arxiv.org/abs/2407.00706)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository accompanies the article:

* **Title**: Sum-of-norms regularized Nonnegative Matrix Factorization
* **Authors**: Andersen Ang, Waqas Bin Hamed, Hans De Sterck
* **arXiv preprint**: [arXiv:2407.00706](https://arxiv.org/abs/2407.00706)

## Overview

This repository provides the implementation and experimental code for the paper "Sum-of-norms regularized Nonnegative Matrix Factorization". 

The method addresses a fundamental challenge in NMF: automatically determining the correct rank (number of components) without prior knowledge. **SONNMF** uses sum-of-norms (SON) regularization to encourage pairwise similarity between factor columns, enabling automatic rank discovery by starting with an overestimated rank and reducing it during optimization.

The repository includes:
- Core SONNMF algorithm implementation
- Jupyter notebooks reproducing all paper experiments
- Datasets used in the evaluation (synthetic, hyperspectral, video)
- Comparison with standard NMF

**Note**: The `main` branch was refactored to improve the code readability by adding comments, restructing the code, reducing repeat code. Data files were earlier stored in `.npz`, now they are in HDF5 (`.h5`) so the data can be accessed in other languages including MATLAB. However, the algorithms and experiments are not affected by this. 

The `article` branch still stores the old version of the code. 

## Repository Structure

```
sonnmf/
├── sonnmf/                 # Core SONNMF implementation
│   ├── core/               # Modern implementation
│   ├── legacy/             # Legacy implementation for comparison
│   ├── numba/              # Optimized Numba implementation
│   └── utils.py            # Utility functions
├── notebooks/              # Jupyter notebooks reproducing paper results
│   ├── synthetic_experiments.ipynb     # Synthetic data experiments
│   ├── urban_experiments.ipynb         # Urban hyperspectral data
│   ├── jasper_experiments.ipynb        # Jasper Ridge hyperspectral data
│   ├── swimmer_experiments.ipynb       # Swimmer video data
│   └── utils.py                        # Shared notebook utilities
├── datasets/               # Dataset files (HDF5 format)
├── archive/                # Archived results and legacy code
└── images/                 # Generated figures and visualizations
```

## Installation

### Prerequisites
- Python 3.7 or higher
- NumPy, SciPy, scikit-learn
- matplotlib, h5py, Pillow
- Jupyter (for notebooks)

### Quick Install

1. **Clone the repository:**
   ```bash
   git clone https://github.com/waqasbinhamed/sonnmf.git
   cd sonnmf
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```
    This step is optional is not required for running notebooks.
## Usage

### Basic Example

```python
import numpy as np
from sonnmf.core.main import sonnmf

# Load your data matrix M (features × samples)
M = np.random.rand(100, 200)  # Example data

# Initialize factors
m, n = M.shape
rank = 20  # Initial rank estimate
W_init = np.random.rand(m, rank)
H_init = np.random.rand(rank, n)

# Run SONNMF
W, H, fscores, gscores, hscores, total_scores = sonnmf(
    M, W_init, H_init,
    lam=1e3,           # Sum-of-norms regularization parameter
    gamma=1e3,         # Total variation regularization parameter
    itermax=1000,      # Maximum iterations
    early_stop=True,   # Enable early stopping
    verbose=True       # Show progress
)

# W contains the basis vectors
# H contains the coefficients
```

### Reproducing Paper Results

The repository includes comprehensive Jupyter notebooks that reproduce all experiments from the paper:

```bash
# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook

# Run the experiment notebooks:
# - synthetic_experiments.ipynb: Synthetic data comparisons
# - urban_experiments.ipynb: Urban hyperspectral unmixing
# - jasper_experiments.ipynb: Jasper Ridge hyperspectral analysis  
# - swimmer_experiments.ipynb: Swimmer image analysis
```

## Experiments

 1. Synthetic data (`synthetic_experiments.ipynb`) compares the algorithm against standard NMF on a toy example. It also compares the proximal averaging algorithm against other solvers for the W sub-problem.

 2. Jasper Ridge (`jasper_experiments.ipynb`) and Urban (`urban_experiments.ipynb`) demonstrate the SONNMF's application to hyperspectral unmixing.

 3. Swimmer dataset consists of 256 figures with each 20-by-11
pixel of a skeleton body “swimming". The notebook (`swimmer_experiments.ipynb`) applies NMF on this dataset.


## Algorithm Details

SONNMF solves the optimization problem:

```
min_{W≥0,H≥0} 0.5||M - WH||²_F + λ·g(W) + γ·h(W) + i(H)
```

Where:
- `g(W)` is the sum-of-norms regularization term
- `h(W)` is the non-negactivity constraint on W
- `i(H)`  is the non-negactivity constraint on H
- `λ, γ` are regularization parameters

The algorithm uses:
- **Block Coordinate Descent**: Alternating optimization of W and H
- **Proximal Averaging**: Efficient solver for the W-subproblem
- **Projected Gradient**: Fast updates for the H-subproblem

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{ang2024sumofnormsregularizednonnegativematrix,
      title={Sum-of-norms regularized Nonnegative Matrix Factorization}, 
      author={Andersen Ang and Waqas Bin Hamed and Hans De Sterck},
      year={2024},
      eprint={2407.00706},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.00706}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the paper or implementation, contact the authors:
- Andersen Ang
- Waqas Bin Hamed (waqasbinhamed@gmail.com)
