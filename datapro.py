import os

import dgl
import numpy as np
import torch
import torch.utils.data.dataset as Dataset

REQUIRED_RAW_FILES = (
    "m_d.csv",
    "m_d_train.csv",
    "m_ss.csv",
    "d_ts.csv",
    "d_ss.csv",
)

TRAIN_ASSOCIATION_FILENAME = "m_d_train.csv"


def read_csv(path):
    """Load a CSV matrix as float32 tensor."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data file is missing: {path}")
    matrix = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return torch.from_numpy(matrix)


def validate_raw_data_dir(data_dir):
    missing = [name for name in REQUIRED_RAW_FILES if not os.path.exists(os.path.join(data_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing required raw data files under {data_dir}: {', '.join(missing)}"
        )


def _validate_matrix_alignment(md_matrix, similarity_matrices):
    if md_matrix.ndim != 2:
        raise ValueError(f"m_d.csv must be a 2D matrix, got shape {tuple(md_matrix.shape)}.")
    num_miRNA, num_disease = md_matrix.shape
    expected_shapes = {
        "m_ss.csv": (num_miRNA, num_miRNA),
        "d_ts.csv": (num_disease, num_disease),
        "d_ss.csv": (num_disease, num_disease),
    }
    for name, matrix in similarity_matrices.items():
        expected = expected_shapes[name]
        if tuple(matrix.shape) != expected:
            raise ValueError(
                f"{name} has shape {tuple(matrix.shape)}, expected {expected} "
                f"to align with m_d.csv shape {(num_miRNA, num_disease)}."
            )
        if not torch.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or infinite values.")

    if not torch.isfinite(md_matrix).all():
        raise ValueError("m_d.csv contains NaN or infinite values.")
    if not torch.all((md_matrix == 0) | (md_matrix == 1)):
        raise ValueError("m_d.csv must contain only binary 0/1 association labels.")


def _gip_similarity(association):
    """Build Gaussian interaction-profile similarity from a training-only matrix."""
    profiles = np.asarray(association, dtype=np.float64)
    squared_norms = np.einsum("ij,ij->i", profiles, profiles)
    gamma = profiles.shape[0] / max(float(squared_norms.sum()), 1e-12)
    distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * profiles.dot(profiles.T)
    np.maximum(distances, 0.0, out=distances)
    similarity = np.exp(-gamma * distances)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def _mirna_functional_similarity(association, disease_similarity):
    """Build miRNA functional similarity using only current graph associations."""
    association = np.asarray(association, dtype=np.int64)
    disease_similarity = np.asarray(disease_similarity, dtype=np.float32)
    disease_sets = [np.flatnonzero(association[index] > 0) for index in range(association.shape[0])]
    similarity = np.eye(association.shape[0], dtype=np.float32)
    for index, diseases_i in enumerate(disease_sets[:-1]):
        if diseases_i.size == 0:
            continue
        for other_index in range(index + 1, len(disease_sets)):
            diseases_j = disease_sets[other_index]
            if diseases_j.size == 0:
                continue
            block = disease_similarity[np.ix_(diseases_i, diseases_j)]
            value = (float(block.max(axis=1).sum()) + float(block.max(axis=0).sum())) / (
                diseases_i.size + diseases_j.size
            )
            similarity[index, other_index] = similarity[other_index, index] = min(max(value, 0.0), 1.0)
    return similarity


def get_edge_index(matrix):
    """Get non-zero indices of a tensor matrix as edge_index in (2, E) shape."""
    non_zero_coords = matrix.nonzero(as_tuple=True)
    return torch.stack(non_zero_coords, dim=0)


def build_weighted_graph(matrix):
    """Build a weighted DGL graph from a dense adjacency/similarity matrix."""
    edge_index = get_edge_index(matrix)
    edge_weight = matrix[edge_index[0], edge_index[1]].float()
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=matrix.shape[0])
    return {
        "data_matrix": matrix.float(),
        "edges": edge_index.long(),
        "edge_weight": edge_weight,
        "graph": graph,
    }


def build_topk_weighted_graph(matrix, topk, lower_topk=0, edge_weight_tau=0.2):
    """Build a symmetric local graph from a Top-k neighbor range.

    By default this builds a nested Top-k graph `(0, topk]`. Passing a
    non-zero `lower_topk` keeps only the band `(lower_topk, topk]`.
    Edge weights are row-wise softmax-normalized before symmetrization.
    """
    matrix = matrix.float()
    num_nodes = matrix.shape[0]
    if num_nodes <= 1:
        return build_weighted_graph(torch.zeros(num_nodes, num_nodes, dtype=torch.float32))

    symmetric = torch.maximum(matrix, matrix.t()).clone()
    symmetric.fill_diagonal_(0.0)
    positive = symmetric.clamp(min=0.0)
    upper_k = min(max(int(topk), 1), num_nodes - 1)
    lower_k = min(max(int(lower_topk), 0), upper_k)
    tau = max(float(edge_weight_tau), 1e-6)

    if upper_k <= lower_k:
        return build_weighted_graph(torch.zeros_like(symmetric))

    top_values, top_indices = torch.topk(positive, k=upper_k, dim=1)
    band_values = top_values[:, lower_k:upper_k]
    band_indices = top_indices[:, lower_k:upper_k]
    positive_mask = band_values > 0

    band_weights = torch.zeros_like(band_values)
    valid_rows = positive_mask.any(dim=1)
    if valid_rows.any():
        scores = (band_values[valid_rows] / tau).masked_fill(~positive_mask[valid_rows], float("-inf"))
        band_weights[valid_rows] = torch.softmax(scores, dim=1)

    directed = torch.zeros_like(symmetric)
    row_indices = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1).expand_as(band_indices)
    directed[row_indices[positive_mask], band_indices[positive_mask]] = band_weights[positive_mask]
    knn_matrix = torch.maximum(directed, directed.t())

    graph_entry = build_weighted_graph(knn_matrix)
    graph_entry["neighbor_range"] = (lower_k + 1, upper_k)
    graph_entry["edge_weight_tau"] = tau
    return graph_entry


def build_multiscale_topk_graphs(matrix, topk_values, edge_weight_tau=0.2, disjoint=False):
    graphs = {}
    previous_topk = 0
    for topk in sorted(int(value) for value in topk_values):
        lower_topk = previous_topk if disjoint else 0
        graphs[int(topk)] = build_topk_weighted_graph(
            matrix,
            topk,
            lower_topk=lower_topk,
            edge_weight_tau=edge_weight_tau,
        )
        previous_topk = int(topk)
    return graphs


def build_bipartite_heterograph(md_matrix):
    """Build a directed heterograph for relation-aware miRNA-disease encoding."""
    num_miRNA, num_disease = md_matrix.shape
    edge_index = get_edge_index(md_matrix)
    src_m = edge_index[0]
    dst_d = edge_index[1]
    graph = dgl.heterograph(
        {
            ("miRNA", "md", "disease"): (src_m, dst_d),
            ("disease", "dm", "miRNA"): (dst_d, src_m),
        },
        num_nodes_dict={"miRNA": num_miRNA, "disease": num_disease},
    )
    edge_weight = torch.ones(src_m.shape[0], dtype=torch.float32)
    return {
        "graph": graph,
        "md_edge_weight": edge_weight,
        "dm_edge_weight": edge_weight.clone(),
        "num_miRNA": num_miRNA,
        "num_disease": num_disease,
    }


def _build_normalized_projection_matrices(md_matrix):
    association = md_matrix.float()
    miRNA_degree = association.sum(dim=1).clamp_min(1.0)
    disease_degree = association.sum(dim=0).clamp_min(1.0)
    miRNA_inv_sqrt = miRNA_degree.pow(-0.5)
    disease_inv_sqrt = disease_degree.pow(-0.5)
    miRNA_inv = miRNA_degree.pow(-1.0)
    disease_inv = disease_degree.pow(-1.0)

    miRNA_projection = torch.matmul(association * disease_inv.unsqueeze(0), association.t())
    miRNA_projection = miRNA_inv_sqrt.unsqueeze(1) * miRNA_projection * miRNA_inv_sqrt.unsqueeze(0)

    disease_projection = torch.matmul(association.t() * miRNA_inv.unsqueeze(0), association)
    disease_projection = disease_inv_sqrt.unsqueeze(1) * disease_projection * disease_inv_sqrt.unsqueeze(0)
    return miRNA_projection, disease_projection


def _topk_sparsify_symmetric(matrix, topk):
    matrix = matrix.float().clone()
    num_nodes = matrix.shape[0]
    if num_nodes == 0:
        return matrix

    matrix.fill_diagonal_(0.0)
    if topk is None or int(topk) <= 0 or num_nodes <= 1:
        matrix = torch.maximum(matrix, matrix.t())
        matrix.fill_diagonal_(0.0)
        return matrix

    k = min(int(topk), num_nodes - 1)
    positive = torch.where(matrix > 0, matrix, torch.zeros_like(matrix))
    top_values, top_indices = torch.topk(positive, k=k, dim=1)
    sparse_matrix = torch.zeros_like(matrix)
    row_indices = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1).expand_as(top_indices)
    keep_mask = top_values > 0
    sparse_matrix[row_indices[keep_mask], top_indices[keep_mask]] = top_values[keep_mask]
    sparse_matrix = torch.maximum(sparse_matrix, sparse_matrix.t())
    sparse_matrix.fill_diagonal_(0.0)
    return sparse_matrix


def _projection_stats(matrix):
    num_nodes = int(matrix.shape[0])
    positive_weights = matrix[matrix > 0]
    edge_count = int(positive_weights.numel())
    density = edge_count / float(max(num_nodes * (num_nodes - 1), 1))
    return {
        "edge_count": float(edge_count),
        "density": float(density),
        "edge_weight_mean": float(positive_weights.mean().item()) if edge_count > 0 else 0.0,
        "edge_weight_max": float(positive_weights.max().item()) if edge_count > 0 else 0.0,
    }


PROJECTION_NEIGHBORS = 20


def build_bipartite_projection_graphs(md_matrix, projection_topk=PROJECTION_NEIGHBORS):
    """Build the fixed normalized, sparse miRNA and disease projection graphs for the global module."""
    miRNA_projection, disease_projection = _build_normalized_projection_matrices(md_matrix)
    miRNA_projection = _topk_sparsify_symmetric(miRNA_projection, projection_topk)
    disease_projection = _topk_sparsify_symmetric(disease_projection, projection_topk)
    miRNA_stats = _projection_stats(miRNA_projection)
    disease_stats = _projection_stats(disease_projection)

    return {
        "miRNA": build_weighted_graph(miRNA_projection),
        "disease": build_weighted_graph(disease_projection),
        "stats": {
            "miRNA_projection_edge_count": miRNA_stats["edge_count"],
            "miRNA_projection_density": miRNA_stats["density"],
            "miRNA_projection_edge_weight_mean": miRNA_stats["edge_weight_mean"],
            "miRNA_projection_edge_weight_max": miRNA_stats["edge_weight_max"],
            "disease_projection_edge_count": disease_stats["edge_count"],
            "disease_projection_density": disease_stats["density"],
            "disease_projection_edge_weight_mean": disease_stats["edge_weight_mean"],
            "disease_projection_edge_weight_max": disease_stats["edge_weight_max"],
            "global_projection_topk": float(projection_topk),
        },
    }


def _load_edge_file(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    return data


def _load_predefined_split(split_dir):
    required = {
        "train_pos": os.path.join(split_dir, "train", "train_pos.csv"),
        "train_neg": os.path.join(split_dir, "train", "train_neg.csv"),
        "test_pos": os.path.join(split_dir, "test", "test_pos.csv"),
        "test_neg": os.path.join(split_dir, "test", "test_neg.csv"),
    }
    missing = [path for path in required.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required split files. This project does not auto-generate train/test splits. "
            f"Missing: {', '.join(missing)}"
        )

    return {
        "train_pos": _load_edge_file(required["train_pos"]),
        "train_neg": _load_edge_file(required["train_neg"]),
        "test_pos": _load_edge_file(required["test_pos"]),
        "test_neg": _load_edge_file(required["test_neg"]),
    }


def _validate_split_against_md(split_data, md_matrix, split_dir):
    positive_keys = ("train_pos", "test_pos")
    negative_keys = ("train_neg", "test_neg")
    num_miRNA, num_disease = md_matrix.shape
    for key, edges in split_data.items():
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"{key} must have shape (N, 2), got {edges.shape}.")
        if edges.size == 0:
            continue
        in_range = (
            (edges[:, 0] >= 0)
            & (edges[:, 0] < num_miRNA)
            & (edges[:, 1] >= 0)
            & (edges[:, 1] < num_disease)
        )
        if not np.all(in_range):
            raise ValueError(
                f"{split_dir} contains out-of-range entity indices in {key}; "
                f"m_d.csv shape is {(num_miRNA, num_disease)}."
            )

    for key in positive_keys:
        edges = split_data[key]
        if edges.size == 0:
            continue
        if not np.all(md_matrix[edges[:, 0], edges[:, 1]] == 1):
            raise ValueError(
                f"{split_dir} contains incompatible positive samples in {key}. "
                "Use split files aligned with the selected dataset's raw/m_d.csv."
            )
    for key in negative_keys:
        edges = split_data[key]
        if edges.size == 0:
            continue
        if not np.all(md_matrix[edges[:, 0], edges[:, 1]] == 0):
            raise ValueError(
                f"{split_dir} contains incompatible negative samples in {key}. "
                "Use split files aligned with the selected dataset's raw/m_d.csv."
            )


def _validate_training_association(data_dir, full_matrix, split_data):
    """Ensure every graph starts without held-out test associations."""
    train_path = os.path.join(data_dir, TRAIN_ASSOCIATION_FILENAME)
    train_matrix = np.loadtxt(train_path, dtype=np.int64, delimiter=",")
    if train_matrix.shape != full_matrix.shape:
        raise ValueError(f"{TRAIN_ASSOCIATION_FILENAME} shape does not match m_d.csv.")
    if not np.all((train_matrix == 0) | (train_matrix == 1)):
        raise ValueError(f"{TRAIN_ASSOCIATION_FILENAME} must contain only binary 0/1 values.")
    if np.any(train_matrix > full_matrix):
        raise ValueError(f"{TRAIN_ASSOCIATION_FILENAME} contains associations absent from m_d.csv.")

    train_pos = split_data["train_pos"]
    test_pos = split_data["test_pos"]
    if not np.all(train_matrix[train_pos[:, 0], train_pos[:, 1]] == 1):
        raise ValueError(f"{TRAIN_ASSOCIATION_FILENAME} is missing a training positive association.")
    if np.any(train_matrix[test_pos[:, 0], test_pos[:, 1]] != 0):
        raise ValueError(f"{TRAIN_ASSOCIATION_FILENAME} contains a held-out test association.")

    expected = full_matrix.copy()
    expected[test_pos[:, 0], test_pos[:, 1]] = 0
    if not np.array_equal(train_matrix, expected):
        raise ValueError(
            f"{TRAIN_ASSOCIATION_FILENAME} must equal m_d.csv with exactly the held-out test positives removed."
        )


def loading_data(param):
    """
    Load miRNA-disease association matrix and the fixed train/test split.
    """
    validate_raw_data_dir(param.data_dir)
    md_path = os.path.join(param.data_dir, "m_d.csv")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Required association matrix is missing: {md_path}")
    md_values = np.loadtxt(md_path, dtype=np.float32, delimiter=",")
    if not np.all((md_values == 0) | (md_values == 1)):
        raise ValueError(f"Association matrix must contain only binary 0/1 values: {md_path}")
    md_matrix = md_values.astype(np.int64)

    split_data = _load_predefined_split(param.split_dir)
    _validate_split_against_md(split_data, md_matrix, param.split_dir)
    _validate_training_association(param.data_dir, md_matrix, split_data)

    train_edges = np.vstack((split_data["train_pos"], split_data["train_neg"])).astype(np.int64)
    test_edges = np.vstack((split_data["test_pos"], split_data["test_neg"])).astype(np.int64)

    train_true_label = np.hstack(
        (
            np.ones(split_data["train_pos"].shape[0], dtype=np.float32),
            np.zeros(split_data["train_neg"].shape[0], dtype=np.float32),
        )
    )
    test_true_label = np.hstack(
        (
            np.ones(split_data["test_pos"].shape[0], dtype=np.float32),
            np.zeros(split_data["test_neg"].shape[0], dtype=np.float32),
        )
    )

    param.num_miRNA, param.num_disease = md_matrix.shape

    return {
        "train_Edges": train_edges,
        "train_Labels": train_true_label,
        "test_Edges": test_edges,
        "test_Labels": test_true_label,
        "true_md": md_matrix,
        "train_pos_edges": split_data["train_pos"],
        "train_neg_edges": split_data["train_neg"],
        "test_pos_edges": split_data["test_pos"],
        "test_neg_edges": split_data["test_neg"],
    }


def Simdata_pro(param, excluded_positive_edges=None):
    """
    Load all similarity/association matrices and construct weighted DGL graphs.
    """
    validate_raw_data_dir(param.data_dir)
    dataset = {}

    mm_s_matrix = read_csv(os.path.join(param.data_dir, "m_ss.csv"))
    dd_t_matrix = read_csv(os.path.join(param.data_dir, "d_ts.csv"))
    dd_s_matrix = read_csv(os.path.join(param.data_dir, "d_ss.csv"))
    label_md_matrix = read_csv(os.path.join(param.data_dir, "m_d.csv"))
    graph_md_path = os.path.join(param.data_dir, TRAIN_ASSOCIATION_FILENAME)
    md_matrix = read_csv(graph_md_path)
    if tuple(label_md_matrix.shape) != tuple(md_matrix.shape):
        raise ValueError(
            f"{os.path.basename(graph_md_path)} has shape {tuple(md_matrix.shape)}, "
            f"but m_d.csv has shape {tuple(label_md_matrix.shape)}."
        )
    if not torch.all((md_matrix == 0) | (md_matrix == 1)):
        raise ValueError(f"{graph_md_path} must contain only binary 0/1 values.")
    if torch.any(md_matrix > label_md_matrix):
        raise ValueError(
            f"{graph_md_path} contains positive edges absent from the full label matrix m_d.csv."
        )
    if excluded_positive_edges is not None:
        edges = torch.as_tensor(excluded_positive_edges, dtype=torch.long)
        if edges.numel():
            if edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError("excluded_positive_edges must have shape (N, 2).")
            if not torch.all(md_matrix[edges[:, 0], edges[:, 1]] == 1):
                raise ValueError("A fold exclusion is not a positive association in m_d_train.csv.")
            md_matrix = md_matrix.clone()
            md_matrix[edges[:, 0], edges[:, 1]] = 0

    association = md_matrix.cpu().numpy()
    mm_f_matrix = torch.from_numpy(_mirna_functional_similarity(association, dd_s_matrix.cpu().numpy()))
    mm_g_matrix = torch.from_numpy(_gip_similarity(association))
    dd_g_matrix = torch.from_numpy(_gip_similarity(association.T))
    _validate_matrix_alignment(
        md_matrix,
        {
            "m_ss.csv": mm_s_matrix,
            "d_ts.csv": dd_t_matrix,
            "d_ss.csv": dd_s_matrix,
        },
    )

    param.num_miRNA, param.num_disease = md_matrix.shape

    dataset["mm_s"] = build_weighted_graph(mm_s_matrix)
    dataset["mm_f"] = build_weighted_graph(mm_f_matrix)
    dataset["mm_g"] = build_weighted_graph(mm_g_matrix)
    dataset["dd_t"] = build_weighted_graph(dd_t_matrix)
    dataset["dd_s"] = build_weighted_graph(dd_s_matrix)
    dataset["dd_g"] = build_weighted_graph(dd_g_matrix)
    local_scale_values = tuple(int(value) for value in param.local_multiscale_topk)
    local_edge_weight_tau = float(getattr(param, "local_edge_weight_tau", 0.2))
    projection_topk = int(getattr(param, "global_projection_topk", PROJECTION_NEIGHBORS))

    dataset["local_graphs"] = {
        "miRNA": {
            "mm_f": build_multiscale_topk_graphs(
                mm_f_matrix,
                local_scale_values,
                edge_weight_tau=local_edge_weight_tau,
                disjoint=False,
            ),
            "mm_s": build_multiscale_topk_graphs(
                mm_s_matrix,
                local_scale_values,
                edge_weight_tau=local_edge_weight_tau,
                disjoint=False,
            ),
            "mm_g": build_multiscale_topk_graphs(
                mm_g_matrix,
                local_scale_values,
                edge_weight_tau=local_edge_weight_tau,
                disjoint=False,
            ),
        },
        "disease": {
            "dd_t": build_multiscale_topk_graphs(
                dd_t_matrix,
                local_scale_values,
                edge_weight_tau=local_edge_weight_tau,
                disjoint=False,
            ),
            "dd_s": build_multiscale_topk_graphs(
                dd_s_matrix,
                local_scale_values,
                edge_weight_tau=local_edge_weight_tau,
                disjoint=False,
            ),
            "dd_g": build_multiscale_topk_graphs(
                dd_g_matrix,
                local_scale_values,
                edge_weight_tau=local_edge_weight_tau,
                disjoint=False,
            ),
        },
        "scale_values": list(local_scale_values),
        "edge_weight_tau": local_edge_weight_tau,
    }
    dataset["m_d_hetero"] = build_bipartite_heterograph(md_matrix)
    dataset["bipartite_projection_graphs"] = build_bipartite_projection_graphs(
        md_matrix,
        projection_topk=projection_topk,
    )
    dataset["meta"] = {
        "num_miRNA": param.num_miRNA,
        "num_disease": param.num_disease,
        "global_projection_topk": projection_topk,
        "graph_association_file": os.path.basename(graph_md_path),
        "graph_positive_count": int(md_matrix.sum().item()),
        "association_derived_similarities": "recomputed_from_current_graph",
    }

    return dataset

class CVEdgeDataset(Dataset.Dataset):
    """Dataset of candidate miRNA-disease edges and labels."""

    def __init__(self, edges, labels):
        self.data = torch.as_tensor(edges, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
