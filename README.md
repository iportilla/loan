# Loan Approval Assignment: Python Setup

This folder holds the BAIM 3220 Kaggle assignment for the
[3220 Loan Approval](https://www.kaggle.com/competitions/3220-loan-approval)
competition. **Start with [`Loan-README.md`](./Loan-README.md).** You can work entirely in a Kaggle notebook, where every
library is already installed. This page is for working on your own laptop instead.

## What's in this folder

| File | For | Purpose |
|---|---|---|
| [`Loan-README.md`](./Loan-README.md) | Students | Step-by-step assignment guide |
| [`loan-starter.ipynb`](./loan-starter.ipynb) | Students | Starter notebook with `# TODO` cells to fill in |
| `requirements.txt`, `environment.yml` | Students | Local Python environment |

## Prerequisites

- Conda (Miniforge, Miniconda, or Anaconda)
- VS Code with the Python and Jupyter extensions, if you plan to use VS Code
- The competition data from Kaggle

## Create the environment

Run these commands from this folder:

```bash
conda env create -f environment.yml
conda activate loan-3220
```

The Conda environment installs the packages listed in `requirements.txt`
(pandas, NumPy, scikit-learn, matplotlib, seaborn, and Jupyter). To confirm
the key libraries can be imported:

```bash
python -c "import pandas as pd, sklearn; print(f'pandas {pd.__version__}; scikit-learn {sklearn.__version__}')"
```

You should see pandas 2.2 or later and scikit-learn 1.5 or later.

## Use the notebook in VS Code

1. Open this folder in VS Code.
2. Open `loan-starter.ipynb`.
3. Select the `loan-3220` Python environment as the notebook kernel.
4. Run the notebook cells from the top.

If the environment is not listed as a kernel, activate it and register it:

```bash
python -m ipykernel install --user --name loan-3220 --display-name "Python (loan-3220)"
```

## Add the Kaggle data

Join the competition, open the **Data** tab, and download `train.csv`,
`test.csv`, and `sample_submission.csv`. Put all three files in this folder,
next to the notebook. The notebook checks these locations, in order:

1. `/kaggle/input/competitions/3220-loan-approval/` (inside a Kaggle notebook)
2. The notebook's own folder
3. The folder above it

It writes predictions to `submission.csv` in the notebook's folder. Upload
that file on the competition page with **Submit Prediction**.
`sample_submission.csv` is an example of the format, not a replacement for
`test.csv`.

## Recreate or update the environment

To create a fresh environment after changing the dependency files:

```bash
conda env create -f environment.yml
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

Every package here is pure Python or has wheels for Windows, macOS (Intel and
Apple Silicon), and Linux, so the same files work on any student laptop.
