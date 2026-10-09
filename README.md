# PGLR-MDA

Code and data for PGLR-MDA.

## Environment

- Python 3.10+
- CUDA 12.8
- PyTorch 2.8.0+cu128
- DGL 2.5

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
pip install -r requirements.txt
```

## Usage

Run from the repository root:

```bash
python main.py
```

The script runs five-fold cross-validation and evaluation on the independent test set. Results are saved to `results/<timestamp>/`.

## Data

The `data/` directory contains association and similarity matrices, entity mappings, and predefined train/test splits. `m_d_train.csv` excludes the held-out test positives. Fold-specific graphs and association-derived similarities are built using the corresponding training data.
