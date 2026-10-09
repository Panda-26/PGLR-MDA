import csv
import os
from datetime import datetime, timezone

TRAIN_HISTORY_FIELDS = [
    "version",
    "model_name",
    "note",
    "mode",
    "fold",
    "epoch",
    "train_loss",
    "valid_auc",
    "valid_aupr",
    "gate_reg_loss",
    "gate_mean",
    "gate_std",
    "local_reliability_mean",
    "global_reliability_mean",
    "branch_disagreement_mean",
    "prob_gap_mean",
    "repr_gap_mean",
    "local_scale_weight_mean",
    "miRNA_view_weight_mean",
    "disease_view_weight_mean",
    "miRNA_projection_weight_mean",
    "disease_projection_weight_mean",
    "miRNA_projection_edge_count",
    "miRNA_projection_density",
    "miRNA_projection_edge_weight_mean",
    "miRNA_projection_edge_weight_max",
    "disease_projection_edge_count",
    "disease_projection_density",
    "disease_projection_edge_weight_mean",
    "disease_projection_edge_weight_max",
    "is_best_epoch",
    "epoch_time_sec",
    "learning_rate",
    "batch_size",
    "seed",
    "timestamp",
]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def local_timestamp_tag():
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def allocate_version_dir(exp_root, experiment_name=None):
    exp_root = os.path.abspath(exp_root)
    _ensure_dir(exp_root)

    experiment_name = experiment_name or "run"
    timestamp_tag = local_timestamp_tag()
    experiment_id = timestamp_tag
    if os.path.exists(os.path.join(exp_root, experiment_id)):
        suffix = 1
        while True:
            candidate = f"{experiment_id}__r{suffix:02d}"
            if not os.path.exists(os.path.join(exp_root, candidate)):
                experiment_id = candidate
                break
            suffix += 1

    experiment_dir = os.path.join(exp_root, experiment_id)
    if os.path.exists(experiment_dir):
        raise FileExistsError(f"Experiment directory already exists: {experiment_dir}")

    paths = {
        "exp_root": exp_root,
        "version": experiment_id,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "timestamp_tag": timestamp_tag,
        "experiment_dir": experiment_dir,
        "metrics_dir": experiment_dir,
        "checkpoints_dir": experiment_dir,
    }
    paths["train_history_path"] = os.path.join(paths["metrics_dir"], "train_history.csv")
    paths["valid_metrics_path"] = os.path.join(paths["metrics_dir"], "fold_metrics.csv")
    paths["valid_summary_path"] = os.path.join(paths["metrics_dir"], "valid_summary.csv")
    paths["test_metrics_path"] = os.path.join(paths["metrics_dir"], "test_metrics.csv")

    os.makedirs(experiment_dir, exist_ok=False)

    return paths


def init_train_history_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_HISTORY_FIELDS)
        writer.writeheader()


def append_train_history_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_HISTORY_FIELDS)
        writer.writerow(row)
