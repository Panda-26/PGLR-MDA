import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from experiment_archive import (
    TRAIN_HISTORY_FIELDS,
    append_train_history_row,
    utc_now_iso,
)
from sklearn import metrics
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from datapro import CVEdgeDataset, Simdata_pro
from model import EmbeddingD, EmbeddingM, PGLRMDA

METRIC_NAMES = ["auc", "aupr", "accuracy", "f1", "recall", "precision"]
AUX_STAT_NAMES = [
    "gate_mean",
    "gate_std",
    "gate_reg_loss",
    "local_reliability_mean",
    "global_reliability_mean",
    "branch_disagreement_mean",
    "prob_gap_mean",
    "repr_gap_mean",
    "miRNA_projection_edge_count",
    "miRNA_projection_density",
    "miRNA_projection_edge_weight_mean",
    "miRNA_projection_edge_weight_max",
    "disease_projection_edge_count",
    "disease_projection_density",
    "disease_projection_edge_weight_mean",
    "disease_projection_edge_weight_max",
]
DIAGNOSTIC_NAMES = [
    "local_scale_weight_mean",
    "miRNA_view_weight_mean",
    "disease_view_weight_mean",
    "miRNA_projection_weight_mean",
    "disease_projection_weight_mean",
]
PROJECTION_GRAPH_STAT_NAMES = [
    "miRNA_projection_edge_count",
    "miRNA_projection_density",
    "miRNA_projection_edge_weight_mean",
    "miRNA_projection_edge_weight_max",
    "disease_projection_edge_count",
    "disease_projection_density",
    "disease_projection_edge_weight_mean",
    "disease_projection_edge_weight_max",
]


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(scores, labels):
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(recall_curve, precision_curve)
    pred_labels = (scores >= 0.5).astype(np.int64)

    return {
        "auc": auc,
        "aupr": aupr,
        "accuracy": metrics.accuracy_score(labels, pred_labels),
        "f1": metrics.f1_score(labels, pred_labels, zero_division=0),
        "recall": metrics.recall_score(labels, pred_labels, zero_division=0),
        "precision": metrics.precision_score(labels, pred_labels, zero_division=0),
    }


def print_metrics(title, metric_values):
    print(
        f"{title}: "
        f"AUC {metric_values['auc']:.4f} | "
        f"AUPR {metric_values['aupr']:.4f} | "
        f"ACC {metric_values['accuracy']:.4f} | "
        f"F1 {metric_values['f1']:.4f} | "
        f"Recall {metric_values['recall']:.4f} | "
        f"Precision {metric_values['precision']:.4f}"
    )


def _new_aux_tracker():
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "local_reliability_sum": 0.0,
        "global_reliability_sum": 0.0,
        "branch_disagreement_sum": 0.0,
        "prob_gap_sum": 0.0,
        "repr_gap_sum": 0.0,
        "gate_reg_loss_sum": 0.0,
        "projection_graph_stat_sums": {name: 0.0 for name in PROJECTION_GRAPH_STAT_NAMES},
        "local_scale_weight_sum": None,
        "miRNA_view_weight_sum": None,
        "disease_view_weight_sum": None,
        "miRNA_projection_weight_sum": None,
        "disease_projection_weight_sum": None,
    }


def _update_vector_tracker(tracker, tracker_key, values, weight):
    if values is None:
        return
    vector = values.detach().reshape(-1).float().cpu().numpy()
    if tracker[tracker_key] is None:
        tracker[tracker_key] = np.zeros_like(vector, dtype=np.float64)
    tracker[tracker_key] += vector * float(weight)


def _update_aux_tracker(tracker, forward_stats):
    gate_values = forward_stats.get("gate_values")
    if gate_values is None:
        return

    values = gate_values.detach().reshape(-1).float().cpu()
    tracker["count"] += values.numel()
    tracker["sum"] += float(values.sum().item())
    tracker["sum_sq"] += float((values * values).sum().item())

    local_reliability = forward_stats.get("local_reliability_values")
    if local_reliability is not None:
        tracker["local_reliability_sum"] += float(local_reliability.detach().reshape(-1).float().cpu().sum().item())

    global_reliability = forward_stats.get("global_reliability_values")
    if global_reliability is not None:
        tracker["global_reliability_sum"] += float(global_reliability.detach().reshape(-1).float().cpu().sum().item())

    branch_disagreement = forward_stats.get("branch_disagreement_values")
    if branch_disagreement is not None:
        tracker["branch_disagreement_sum"] += float(
            branch_disagreement.detach().reshape(-1).float().cpu().sum().item()
        )

    prob_gap = forward_stats.get("prob_gap_values")
    if prob_gap is not None:
        tracker["prob_gap_sum"] += float(prob_gap.detach().reshape(-1).float().cpu().sum().item())

    repr_gap = forward_stats.get("repr_gap_values")
    if repr_gap is not None:
        tracker["repr_gap_sum"] += float(repr_gap.detach().reshape(-1).float().cpu().sum().item())

    gate_reg_loss = forward_stats.get("gate_reg_loss")
    if gate_reg_loss is not None:
        tracker["gate_reg_loss_sum"] += float(gate_reg_loss.detach().item()) * values.numel()

    for stat_name in PROJECTION_GRAPH_STAT_NAMES:
        stat_value = forward_stats.get(stat_name)
        if stat_value is not None:
            tracker["projection_graph_stat_sums"][stat_name] += float(stat_value) * values.numel()

    _update_vector_tracker(
        tracker,
        "local_scale_weight_sum",
        forward_stats.get("local_scale_weight_mean_tensor"),
        values.numel(),
    )
    _update_vector_tracker(
        tracker,
        "miRNA_view_weight_sum",
        forward_stats.get("miRNA_view_weight_mean_tensor"),
        values.numel(),
    )
    _update_vector_tracker(
        tracker,
        "disease_view_weight_sum",
        forward_stats.get("disease_view_weight_mean_tensor"),
        values.numel(),
    )
    _update_vector_tracker(
        tracker,
        "miRNA_projection_weight_sum",
        forward_stats.get("miRNA_projection_weight_mean_tensor"),
        values.numel(),
    )
    _update_vector_tracker(
        tracker,
        "disease_projection_weight_sum",
        forward_stats.get("disease_projection_weight_mean_tensor"),
        values.numel(),
    )


def _finalize_aux_tracker(tracker):
    def _finalize_vector(name):
        values = tracker[name]
        if tracker["count"] == 0 or values is None:
            return []
        return (values / tracker["count"]).tolist()

    if tracker["count"] == 0:
        empty_stats = {
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "gate_reg_loss": 0.0,
            "local_reliability_mean": 0.0,
            "global_reliability_mean": 0.0,
            "branch_disagreement_mean": 0.0,
            "prob_gap_mean": 0.0,
            "repr_gap_mean": 0.0,
            "local_scale_weight_mean": [],
            "miRNA_view_weight_mean": [],
            "disease_view_weight_mean": [],
            "miRNA_projection_weight_mean": [],
            "disease_projection_weight_mean": [],
        }
        empty_stats.update({name: 0.0 for name in PROJECTION_GRAPH_STAT_NAMES})
        return empty_stats

    gate_mean = tracker["sum"] / tracker["count"]
    gate_var = max(tracker["sum_sq"] / tracker["count"] - gate_mean * gate_mean, 0.0)
    finalized = {
        "gate_mean": float(gate_mean),
        "gate_std": float(gate_var ** 0.5),
        "gate_reg_loss": float(tracker["gate_reg_loss_sum"] / tracker["count"]),
        "local_reliability_mean": float(tracker["local_reliability_sum"] / tracker["count"]),
        "global_reliability_mean": float(tracker["global_reliability_sum"] / tracker["count"]),
        "branch_disagreement_mean": float(tracker["branch_disagreement_sum"] / tracker["count"]),
        "prob_gap_mean": float(tracker["prob_gap_sum"] / tracker["count"]),
        "repr_gap_mean": float(tracker["repr_gap_sum"] / tracker["count"]),
        "local_scale_weight_mean": _finalize_vector("local_scale_weight_sum"),
        "miRNA_view_weight_mean": _finalize_vector("miRNA_view_weight_sum"),
        "disease_view_weight_mean": _finalize_vector("disease_view_weight_sum"),
        "miRNA_projection_weight_mean": _finalize_vector("miRNA_projection_weight_sum"),
        "disease_projection_weight_mean": _finalize_vector("disease_projection_weight_sum"),
    }
    finalized.update(
        {
            name: float(tracker["projection_graph_stat_sums"][name] / tracker["count"])
            for name in PROJECTION_GRAPH_STAT_NAMES
        }
    )
    return finalized


def _select_best_fold(fold_metrics):
    return sorted(fold_metrics, key=lambda item: (-item["aupr"], -item["auc"], item["fold"]))[0]


def _is_better_metrics(candidate_metrics, best_metrics):
    eps = 1e-12
    if best_metrics is None:
        return True
    if candidate_metrics["aupr"] > best_metrics["aupr"] + eps:
        return True
    if abs(candidate_metrics["aupr"] - best_metrics["aupr"]) <= eps and candidate_metrics["auc"] > best_metrics["auc"] + eps:
        return True
    return False


def _prepare_sim_data_for_device(sim_data, device):
    def _move_in_place(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "graph" and hasattr(item, "to"):
                    item_device = getattr(item, "device", None)
                    if item_device is None or item_device != device:
                        value[key] = item.to(device)
                elif torch.is_tensor(item):
                    if item.device != device:
                        value[key] = item.to(device, non_blocking=True)
                elif isinstance(item, (dict, list)):
                    _move_in_place(item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if torch.is_tensor(item):
                    if item.device != device:
                        value[index] = item.to(device, non_blocking=True)
                elif isinstance(item, (dict, list)):
                    _move_in_place(item)

    _move_in_place(sim_data)
    return sim_data


def build_model(param):
    model = PGLRMDA(param, EmbeddingM(param), EmbeddingD(param))
    return model.to(param.device)


def build_loader(edges, labels, batch_size, shuffle):
    dataset = CVEdgeDataset(edges, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, sim_data, data_loader, optimizer, device):
    model.train()
    criterion = torch.nn.BCELoss()
    total_loss = 0.0
    aux_tracker = _new_aux_tracker()

    for data, label in data_loader:
        train_data = data.to(device, non_blocking=True)
        true_label = label.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred_score = model(sim_data, train_data)
        forward_stats = model.get_latest_forward_stats()
        gate_reg_loss = forward_stats.get("gate_reg_loss", pred_score.new_zeros(()))
        if isinstance(gate_reg_loss, float):
            gate_reg_loss = pred_score.new_tensor(gate_reg_loss)
        loss = criterion(pred_score, true_label) + gate_reg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _update_aux_tracker(aux_tracker, forward_stats)

    return total_loss / max(len(data_loader), 1), _finalize_aux_tracker(aux_tracker)


def evaluate(model, sim_data, data_loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    aux_tracker = _new_aux_tracker()

    with torch.no_grad():
        cached_state = model.prepare_batch_invariant_state(sim_data)
        for data, label in data_loader:
            batch_data = data.to(device, non_blocking=True)
            pred_score = model.predict_from_cached_state(cached_state, batch_data)
            forward_stats = model.get_latest_forward_stats()
            all_scores.append(pred_score.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())
            _update_aux_tracker(aux_tracker, forward_stats)

    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    return (
        scores,
        labels,
        calculate_metrics(scores, labels),
        _finalize_aux_tracker(aux_tracker),
        [],
    )


def _metric_summary_rows(fold_metrics, metric_names):
    df = pd.DataFrame(fold_metrics)
    rows = []
    for name in metric_names:
        values = pd.to_numeric(df[name], errors="coerce").dropna()
        rows.append(
            {
                "metric": name,
                "mean": float(values.mean()) if len(values) else 0.0,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
        )
    return rows


def _write_valid_metrics(fold_metrics_path, valid_summary_path, fold_metrics):
    df = pd.DataFrame(fold_metrics)
    df.to_csv(fold_metrics_path, index=False)
    pd.DataFrame(_metric_summary_rows(fold_metrics, METRIC_NAMES)).to_csv(valid_summary_path, index=False)


def _summary_dict(fold_metrics, metric_names):
    return {
        row["metric"]: {"mean": row["mean"], "std": row["std"], "n": row["n"]}
        for row in _metric_summary_rows(fold_metrics, metric_names)
    }


def _state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _save_fold_checkpoint(param, fold_id, state_dict):
    checkpoint_dir = getattr(param, "checkpoints_dir", "")
    if not checkpoint_dir or state_dict is None:
        return ""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"fold_{int(fold_id)}_best_state.pt")
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def _average_vector_stats(aux_stats_list, name):
    vectors = [stats.get(name, []) for stats in aux_stats_list]
    vectors = [np.asarray(vector, dtype=np.float64) for vector in vectors if len(vector) > 0]
    if not vectors:
        return []
    min_size = min(vector.size for vector in vectors)
    if min_size == 0:
        return []
    stacked = np.stack([vector[:min_size] for vector in vectors], axis=0)
    return stacked.mean(axis=0).tolist()


def _average_aux_stats(aux_stats_list):
    if not aux_stats_list:
        empty_stats = {name: 0.0 for name in AUX_STAT_NAMES}
        empty_stats.update({name: [] for name in DIAGNOSTIC_NAMES})
        return empty_stats

    averaged = {}
    for name in AUX_STAT_NAMES:
        values = [float(stats.get(name, 0.0)) for stats in aux_stats_list]
        averaged[name] = float(np.mean(values)) if values else 0.0
    for name in DIAGNOSTIC_NAMES:
        averaged[name] = _average_vector_stats(aux_stats_list, name)
    return averaged


def _write_test_metrics(path, param, test_metrics, test_aux_stats, checkpoint_selector):
    row = {
        "version": param.experiment_version,
        "model_name": param.model_name,
        "note": param.note,
        "checkpoint_selector": checkpoint_selector,
        **{name: float(test_metrics[name]) for name in METRIC_NAMES},
        **{name: float(test_aux_stats[name]) for name in AUX_STAT_NAMES},
        **{name: json.dumps(test_aux_stats.get(name, [])) for name in DIAGNOSTIC_NAMES},
    }
    pd.DataFrame([row]).to_csv(path, index=False)


def _serialize_diagnostics(aux_stats):
    return {
        name: aux_stats.get(name, [])
        for name in DIAGNOSTIC_NAMES
    }


def train_and_validate(split_data, param):
    train_edges = split_data["train_Edges"]
    train_labels = split_data["train_Labels"]
    kf = KFold(n_splits=param.kfold, shuffle=True, random_state=param.seed)
    fold_metrics = []
    fold_best_states = {}
    fold_graph_exclusions = {}

    for fold_id, (train_index, valid_index) in enumerate(kf.split(train_edges), start=1):
        print(f"################ Fold {fold_id}/{param.kfold} ################")
        model = build_model(param)
        optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate, weight_decay=param.weight_decay)

        edges_train = train_edges[train_index]
        edges_valid = train_edges[valid_index]
        labels_train = train_labels[train_index]
        labels_valid = train_labels[valid_index]
        valid_positive_edges = edges_valid[labels_valid == 1]
        sim_data = _prepare_sim_data_for_device(
            Simdata_pro(param, excluded_positive_edges=valid_positive_edges),
            param.device,
        )

        train_loader = build_loader(edges_train, labels_train, param.batch_size, shuffle=True)
        valid_loader = build_loader(edges_valid, labels_valid, param.batch_size, shuffle=False)
        best_valid_metrics = None
        best_valid_aux_stats = None
        best_state_dict = None
        best_epoch = 0
        epochs_without_improve = 0
        epochs_ran = 0

        for epoch in range(1, param.max_epochs + 1):
            start = time.time()
            avg_loss, train_aux_stats = run_epoch(model, sim_data, train_loader, optimizer, param.device)
            _, _, valid_metrics, valid_aux_stats, _ = evaluate(model, sim_data, valid_loader, param.device)
            is_best_epoch = _is_better_metrics(valid_metrics, best_valid_metrics)
            if is_best_epoch:
                best_valid_metrics = dict(valid_metrics)
                best_valid_aux_stats = dict(valid_aux_stats)
                best_state_dict = _state_dict_to_cpu(model.state_dict())
                best_epoch = epoch
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
            elapsed = time.time() - start
            epochs_ran = epoch
            print(
                f"epoch {epoch}/{param.max_epochs}: loss={avg_loss:.4f}, "
                f"valid_auc={valid_metrics['auc']:.4f}, "
                f"valid_aupr={valid_metrics['aupr']:.4f}, "
                f"gate_mean={train_aux_stats['gate_mean']:.4f}, "
                f"disagreement={train_aux_stats['branch_disagreement_mean']:.4f}, "
                f"local_rel={train_aux_stats['local_reliability_mean']:.4f}, "
                f"global_rel={train_aux_stats['global_reliability_mean']:.4f}, "
                f"scale_weights={json.dumps(train_aux_stats['local_scale_weight_mean'])}, "
                f"projection_weights=[m:{json.dumps(train_aux_stats['miRNA_projection_weight_mean'])}, "
                f"d:{json.dumps(train_aux_stats['disease_projection_weight_mean'])}], "
                f"projection_edges=[m:{train_aux_stats['miRNA_projection_edge_count']:.0f}, "
                f"d:{train_aux_stats['disease_projection_edge_count']:.0f}], "
                f"projection_density=[m:{train_aux_stats['miRNA_projection_density']:.6f}, "
                f"d:{train_aux_stats['disease_projection_density']:.6f}], "
                f"gate_std={train_aux_stats['gate_std']:.4f}, "
                f"time={elapsed:.2f}s"
            )
            append_train_history_row(
                param.train_history_path,
                {
                    field: value
                    for field, value in {
                        "version": param.experiment_version,
                        "model_name": param.model_name,
                        "note": param.note,
                        "fold": fold_id,
                        "epoch": epoch,
                        "train_loss": f"{avg_loss:.6f}",
                        "valid_auc": f"{valid_metrics['auc']:.6f}",
                        "valid_aupr": f"{valid_metrics['aupr']:.6f}",
                        "gate_reg_loss": f"{train_aux_stats['gate_reg_loss']:.6f}",
                        "gate_mean": f"{train_aux_stats['gate_mean']:.6f}",
                        "gate_std": f"{train_aux_stats['gate_std']:.6f}",
                        "local_reliability_mean": f"{train_aux_stats['local_reliability_mean']:.6f}",
                        "global_reliability_mean": f"{train_aux_stats['global_reliability_mean']:.6f}",
                        "branch_disagreement_mean": f"{train_aux_stats['branch_disagreement_mean']:.6f}",
                        "prob_gap_mean": f"{train_aux_stats['prob_gap_mean']:.6f}",
                        "repr_gap_mean": f"{train_aux_stats['repr_gap_mean']:.6f}",
                        "local_scale_weight_mean": json.dumps(train_aux_stats["local_scale_weight_mean"]),
                        "miRNA_view_weight_mean": json.dumps(train_aux_stats["miRNA_view_weight_mean"]),
                        "disease_view_weight_mean": json.dumps(train_aux_stats["disease_view_weight_mean"]),
                        "miRNA_projection_weight_mean": json.dumps(train_aux_stats["miRNA_projection_weight_mean"]),
                        "disease_projection_weight_mean": json.dumps(train_aux_stats["disease_projection_weight_mean"]),
                        "miRNA_projection_edge_count": f"{train_aux_stats['miRNA_projection_edge_count']:.6f}",
                        "miRNA_projection_density": f"{train_aux_stats['miRNA_projection_density']:.6f}",
                        "miRNA_projection_edge_weight_mean": f"{train_aux_stats['miRNA_projection_edge_weight_mean']:.6f}",
                        "miRNA_projection_edge_weight_max": f"{train_aux_stats['miRNA_projection_edge_weight_max']:.6f}",
                        "disease_projection_edge_count": f"{train_aux_stats['disease_projection_edge_count']:.6f}",
                        "disease_projection_density": f"{train_aux_stats['disease_projection_density']:.6f}",
                        "disease_projection_edge_weight_mean": f"{train_aux_stats['disease_projection_edge_weight_mean']:.6f}",
                        "disease_projection_edge_weight_max": f"{train_aux_stats['disease_projection_edge_weight_max']:.6f}",
                        "is_best_epoch": int(is_best_epoch),
                        "epoch_time_sec": f"{elapsed:.4f}",
                        "learning_rate": param.learning_rate,
                        "batch_size": param.batch_size,
                        "seed": param.seed,
                        "timestamp": utc_now_iso(),
                    }.items()
                    if field in TRAIN_HISTORY_FIELDS
                },
            )

            if epoch >= param.min_epochs and epochs_without_improve >= param.early_stop_patience:
                print(
                    f"fold {fold_id} early stopped at epoch {epoch}; "
                    f"best_epoch={best_epoch} best_valid_aupr={best_valid_metrics['aupr']:.4f}"
                )
                break

        print_metrics(f"fold {fold_id} valid(best)", best_valid_metrics)

        checkpoint_path = _save_fold_checkpoint(param, fold_id, best_state_dict)
        fold_best_states[fold_id] = best_state_dict
        fold_graph_exclusions[fold_id] = valid_positive_edges
        fold_metrics.append(
            {
                "version": param.experiment_version,
                "model_name": param.model_name,
                "note": param.note,
                "fold": fold_id,
                "best_epoch": best_epoch,
                "epochs_ran": epochs_ran,
                "stopped_early": int(epochs_ran < param.max_epochs),
                "selected_for_test": False,
                "checkpoint_path": checkpoint_path,
                **best_valid_metrics,
                **best_valid_aux_stats,
                "local_scale_weight_mean": json.dumps(best_valid_aux_stats["local_scale_weight_mean"]),
                "miRNA_view_weight_mean": json.dumps(best_valid_aux_stats["miRNA_view_weight_mean"]),
                "disease_view_weight_mean": json.dumps(best_valid_aux_stats["disease_view_weight_mean"]),
                "miRNA_projection_weight_mean": json.dumps(best_valid_aux_stats["miRNA_projection_weight_mean"]),
                "disease_projection_weight_mean": json.dumps(best_valid_aux_stats["disease_projection_weight_mean"]),
            }
        )

    best_fold_entry = _select_best_fold(fold_metrics)
    best_fold_id = int(best_fold_entry["fold"])
    for fold_metric in fold_metrics:
        fold_metric["selected_for_test"] = True

    print(
        "test checkpoint selection: "
        f"5-fold ensemble averaging; reference best fold {best_fold_id} "
        f"by valid AUPR={best_fold_entry['aupr']:.4f}, AUC={best_fold_entry['auc']:.4f}"
    )
    _write_valid_metrics(param.valid_metrics_path, param.valid_summary_path, fold_metrics)

    metric_summary = _summary_dict(fold_metrics, METRIC_NAMES)
    aux_summary = _summary_dict(fold_metrics, AUX_STAT_NAMES)
    selected_diagnostics = {
        name: json.loads(best_fold_entry[name]) if isinstance(best_fold_entry[name], str) else best_fold_entry[name]
        for name in DIAGNOSTIC_NAMES
    }
    return {
        "fold_metrics": fold_metrics,
        "note": param.note,
        "metric_summary": metric_summary,
        "aux_summary": aux_summary,
        "mean_metrics": {name: float(metric_summary[name]["mean"]) for name in METRIC_NAMES},
        "std_metrics": {name: float(metric_summary[name]["std"]) for name in METRIC_NAMES},
        "best_fold": best_fold_id,
        "best_epoch": int(best_fold_entry["best_epoch"]),
        "best_valid_metrics": {name: float(best_fold_entry[name]) for name in METRIC_NAMES},
        "best_valid_aux_stats": {name: float(best_fold_entry[name]) for name in AUX_STAT_NAMES},
        "best_diagnostics": selected_diagnostics,
        "selection_rule": "highest_valid_aupr_then_auc",
        "valid_metrics_path": param.valid_metrics_path,
        "valid_summary_path": param.valid_summary_path,
        "_best_state_dict": fold_best_states[best_fold_id],
        "_fold_state_dicts": fold_best_states,
        "_fold_graph_exclusions": fold_graph_exclusions,
    }


def run_test(split_data, param, valid_result):
    test_loader = build_loader(split_data["test_Edges"], split_data["test_Labels"], param.batch_size, shuffle=False)
    start = time.time()

    fold_state_dicts = valid_result["_fold_state_dicts"]
    fold_graph_exclusions = valid_result["_fold_graph_exclusions"]
    model = build_model(param)
    fold_scores = []
    fold_aux_stats = []
    reference_labels = None
    for fold_id in sorted(fold_state_dicts):
        sim_data = _prepare_sim_data_for_device(
            Simdata_pro(param, excluded_positive_edges=fold_graph_exclusions[fold_id]),
            param.device,
        )
        model.load_state_dict(fold_state_dicts[fold_id])
        scores, labels, fold_test_metrics, aux_stats, _ = evaluate(
            model,
            sim_data,
            test_loader,
            param.device,
        )
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError("Test labels changed across ensemble fold predictions.")
        fold_scores.append(scores)
        fold_aux_stats.append(aux_stats)
        print_metrics(f"test fold-model {fold_id}", fold_test_metrics)

    mean_scores = np.mean(np.stack(fold_scores, axis=0), axis=0)
    test_metrics = calculate_metrics(mean_scores, reference_labels)
    test_aux_stats = _average_aux_stats(fold_aux_stats)
    checkpoint_selector = f"{len(fold_scores)}_fold_ensemble_mean"

    elapsed = time.time() - start
    print(f"test time={elapsed:.2f}s")
    print_metrics("test", test_metrics)

    _write_test_metrics(
        param.test_metrics_path,
        param,
        test_metrics,
        test_aux_stats,
        checkpoint_selector,
    )

    return {
        "note": param.note,
        "metrics": {name: float(test_metrics[name]) for name in METRIC_NAMES},
        "aux_stats": {name: float(test_aux_stats[name]) for name in AUX_STAT_NAMES},
        "diagnostics": _serialize_diagnostics(test_aux_stats),
        "checkpoint_selector": checkpoint_selector,
        "elapsed_time_sec": float(elapsed),
        "test_metrics_path": param.test_metrics_path,
    }


def run_experiment(split_data, param):
    result = {
        "version": param.experiment_version,
        "model_name": param.model_name,
        "experiment_id": param.experiment_id,
        "experiment_name": param.experiment_name,
        "note": param.note,
        "experiment_dir": param.experiment_dir,
        "valid": None,
        "test": None,
    }

    result["valid"] = train_and_validate(split_data, param)
    result["test"] = run_test(split_data, param, result["valid"])
    result["valid"].pop("_best_state_dict", None)
    result["valid"].pop("_fold_state_dicts", None)
    result["valid"].pop("_fold_graph_exclusions", None)
    return result
