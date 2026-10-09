import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from datapro import loading_data
from experiment_archive import (
    allocate_version_dir,
    init_train_history_csv,
)
from train_runtime import run_experiment, setup_seed


DEFAULT_GATE_REG_WEIGHT = 1e-3


@dataclass
class Config:
    data_dir: str
    split_dir: str
    device: torch.device
    model_name: str = "PGLR-MDA"
    note: str = "PGLR-MDA"
    exp_root: str = "./results"
    seed: int = 42
    kfold: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gate_reg_weight: float = DEFAULT_GATE_REG_WEIGHT
    local_multiscale_topk: tuple[int, ...] = (10, 20, 40)
    local_edge_weight_tau: float = 1.0
    local_encoder_layers: int = 2
    local_encoder_hidden_dim: int = 128
    local_encoder_dropout: float = 0.1
    local_attention_mix: float = 1.0
    global_rel_layers: int = 2
    global_rel_dropout: float = 0.3
    global_projection_topk: int = 10
    global_projection_hidden_dim: int = 128
    max_epochs: int = 40
    min_epochs: int = 12
    early_stop_patience: int = 6
    gcn_layers1: int = 2
    view: int = 3
    fm: int = 128
    fd: int = 128
    outSize: int = 128
    fcDropout: float = 0.1
    num_heads1: int = 2
    num_miRNA: int = 0
    num_disease: int = 0
    experiment_version: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    experiment_dir: str = ""
    metrics_dir: str = ""
    train_history_path: str = ""
    valid_metrics_path: str = ""
    valid_summary_path: str = ""
    test_metrics_path: str = ""
    checkpoints_dir: str = ""


def build_config():
    package_root = Path(__file__).resolve().parent
    data_root = package_root / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Config(
        data_dir=str(data_root / "raw"),
        split_dir=str(data_root / "splits"),
        device=device,
        exp_root=str(package_root / "results"),
    )


def initialize_archive(param):
    archive_paths = allocate_version_dir(param.exp_root, experiment_name=param.note)
    init_train_history_csv(archive_paths["train_history_path"])

    param.exp_root = archive_paths["exp_root"]
    param.experiment_version = archive_paths["version"]
    param.experiment_id = archive_paths["experiment_id"]
    param.experiment_name = archive_paths["experiment_name"]
    param.experiment_dir = archive_paths["experiment_dir"]
    param.metrics_dir = archive_paths["metrics_dir"]
    param.train_history_path = archive_paths["train_history_path"]
    param.valid_metrics_path = archive_paths["valid_metrics_path"]
    param.valid_summary_path = archive_paths["valid_summary_path"]
    param.test_metrics_path = archive_paths["test_metrics_path"]
    param.checkpoints_dir = archive_paths["checkpoints_dir"]

    return archive_paths


def main():
    param = build_config()
    initialize_archive(param)

    error = None
    exit_code = 0

    try:
        print(f"model_name={param.model_name}")
        print(f"device={param.device} version={param.experiment_version}")
        print(f"experiment_dir={param.experiment_dir}")
        print(
            f"data_dir={param.data_dir} split_dir={param.split_dir}"
        )
        print(
            f"local_multiscale_topk={param.local_multiscale_topk} "
            f"global_projection_topk={param.global_projection_topk} "
            f"gate_reg_weight={param.gate_reg_weight} "
            f"global_rel_dropout={param.global_rel_dropout}"
        )
        print(
            f"training_schedule=max_epochs:{param.max_epochs} "
            f"min_epochs:{param.min_epochs} patience:{param.early_stop_patience}"
        )
        print("run_notice=5-fold validation followed by five-model ensemble evaluation on the independent test set")

        setup_seed(param.seed)
        split_data = loading_data(param)
        print(f"num_miRNA={param.num_miRNA} num_disease={param.num_disease}")
        run_experiment(split_data, param)
    except Exception as exc:
        exit_code = 1
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        traceback.print_exc()

    if error is not None:
        print(f"run_failed={error['type']}: {error['message']}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
