## Environment

The environment used CUDA 12.8, PyTorch 2.8.0+cu128 and DGL 2.5. Use the same Python version as the server environment (Python 3.10 or newer), then install the CUDA-enabled PyTorch build before the remaining packages:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
pip install -r requirements.txt
```

## Run the complete model

Run from this directory:

```bash
python main.py
```

The program automatically uses CUDA when it is available, otherwise CPU. It runs five-fold training/validation, then evaluates the averaged five-model prediction on the independent test set. Outputs and checkpoints are saved together in `results/<timestamp>/` inside this project directory, regardless of the working directory. Each run uses a separate directory to preserve earlier results; no seed, model, metrics or checkpoint subdirectories are created.

## Source files

- `main.py`: configuration and execution entry point.
- `model.py`: model architecture and prediction.
- `basicModules.py`: neural network building blocks.
- `datapro.py`: data loading, validation and graph construction.
- `train_runtime.py`: training, checkpoint selection and evaluation.
- `experiment_archive.py`: output directories, configuration and metric recording.

## Data

`data/` contains the full label matrix, three external similarity matrices, `m_d_train.csv`, fixed train/test splits and entity mappings. `m_d_train.csv` is the full association matrix with all 3,397 held-out test positives removed; each cross-validation fold additionally removes that fold's validation positives before graph construction. The association-derived miRNA functional similarity and miRNA/disease GIP similarities are recomputed from this current graph rather than read from a fixed file. The dataset contains 1,104 miRNAs, 850 diseases and 16,988 known positive associations. Training and test sets respectively contain 13,591 and 3,397 positives, each with an equal number of sampled unverified pairs. Sampled negative labels are not experimentally confirmed negative associations.
