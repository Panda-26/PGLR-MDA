import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from dgl.nn.pytorch import EdgeWeightNorm, GATConv, GraphConv, HeteroGraphConv

from basicModules import LayerAtt2, MLP


def _ensure_graph_entry_device(entry, device):
    graph = entry["graph"]
    if graph.device != device:
        graph = graph.to(device)
        entry["graph"] = graph

    edge_weight = entry["edge_weight"]
    if edge_weight.device != device:
        edge_weight = edge_weight.to(device)
        entry["edge_weight"] = edge_weight

    if "edges" in entry and entry["edges"].device != device:
        entry["edges"] = entry["edges"].to(device)

    if "data_matrix" in entry and entry["data_matrix"].device != device:
        entry["data_matrix"] = entry["data_matrix"].to(device)

    return graph, edge_weight


def _ensure_hetero_graph_entry_device(entry, device):
    graph = entry["graph"]
    if graph.device != device:
        graph = graph.to(device)
        entry["graph"] = graph
    return graph


def _binary_entropy(prob):
    prob = prob.clamp(min=1e-6, max=1.0 - 1e-6)
    return -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())


class ReliabilityGateFusion(nn.Module):
    def __init__(self, embed_dim, alpha_min=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha_min = float(alpha_min)
        self.local_reliability = nn.Linear(3, 1)
        self.global_reliability = nn.Linear(3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.local_reliability.weight)
        nn.init.zeros_(self.local_reliability.bias)
        nn.init.xavier_uniform_(self.global_reliability.weight)
        nn.init.zeros_(self.global_reliability.bias)

    def _branch_features(self, pair_embed, logits, prob):
        certainty = 1.0 - (_binary_entropy(prob) / np.log(2.0))
        margin = torch.tanh(torch.abs(logits))
        strength = torch.tanh(torch.norm(pair_embed, dim=1) / np.sqrt(float(self.embed_dim)))
        return torch.stack([certainty, margin, strength], dim=1)

    def forward(self, local_embed, global_embed, local_logits, global_logits):
        local_prob = torch.sigmoid(local_logits)
        global_prob = torch.sigmoid(global_logits)

        local_features = self._branch_features(local_embed, local_logits, local_prob)
        global_features = self._branch_features(global_embed, global_logits, global_prob)
        local_reliability = F.softplus(self.local_reliability(local_features).squeeze(dim=1))
        global_reliability = F.softplus(self.global_reliability(global_features).squeeze(dim=1))

        prob_gap = torch.abs(local_prob - global_prob).clamp(min=0.0, max=1.0)
        repr_gap = (0.5 * (1.0 - F.cosine_similarity(local_embed, global_embed, dim=1, eps=1e-8))).clamp(
            min=0.0,
            max=1.0,
        )
        disagreement = (0.5 * prob_gap + 0.5 * repr_gap).clamp(min=0.0, max=1.0)
        alpha_base = global_reliability / (local_reliability + global_reliability + 1e-8)
        alpha_raw = ((1.0 - disagreement) * 0.5 + disagreement * alpha_base).clamp(min=0.0, max=1.0)
        alpha = self.alpha_min + (1.0 - 2.0 * self.alpha_min) * alpha_raw

        return {
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "local_reliability": local_reliability,
            "global_reliability": global_reliability,
            "prob_gap": prob_gap,
            "repr_gap": repr_gap,
            "disagreement": disagreement,
            "local_prob": local_prob,
            "global_prob": global_prob,
            "alpha_base": alpha_base,
            "local_certainty": local_features[:, 0],
            "global_certainty": global_features[:, 0],
            "local_margin": local_features[:, 1],
            "global_margin": global_features[:, 1],
            "local_strength": local_features[:, 2],
            "global_strength": global_features[:, 2],
        }


class LocalGraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, dropout, num_heads, hidden_dim):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim={input_dim} must be divisible by num_heads={num_heads}")

        head_dim = input_dim // num_heads
        self.layers = nn.ModuleList(
            [
                GATConv(
                    in_feats=input_dim,
                    out_feats=head_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=0.2,
                    allow_zero_in_degree=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.weighted_layers = nn.ModuleList(
            [GraphConv(input_dim, input_dim, norm="none", allow_zero_in_degree=True) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, input_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])

    def forward(self, graph_entry, features):
        graph, edge_weight = _ensure_graph_entry_device(graph_entry, features.device)
        edge_weight = edge_weight.float().clamp(min=0.0)
        hidden = features
        for attn, weighted_conv, attn_norm, ffn, ffn_norm in zip(
            self.layers,
            self.weighted_layers,
            self.attn_norms,
            self.ffn_layers,
            self.ffn_norms,
        ):
            attn_hidden = attn(graph, hidden).flatten(1)
            if graph.num_edges() > 0:
                weighted_hidden = weighted_conv(graph, hidden, edge_weight=edge_weight)
            else:
                weighted_hidden = torch.zeros_like(attn_hidden)
            local_hidden = torch.relu(attn_hidden + weighted_hidden)
            hidden = attn_norm(hidden + self.dropout(local_hidden))
            ffn_hidden = ffn(hidden)
            hidden = ffn_norm(hidden + self.dropout(ffn_hidden))
        return hidden


class MultiViewAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, attention_mix=0.7):
        super().__init__()
        self.attention_mix = float(min(max(attention_mix, 0.0), 1.0))
        self.view_attention = nn.Linear(input_dim, 1)
        self.update = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, base_feature, view_features):
        stacked_views = torch.stack(view_features, dim=1)
        view_weights = torch.softmax(self.view_attention(stacked_views).squeeze(dim=-1), dim=1)
        attention_view = torch.sum(stacked_views * view_weights.unsqueeze(dim=-1), dim=1)
        mean_view = stacked_views.mean(dim=1)
        fused_view = self.attention_mix * attention_view + (1.0 - self.attention_mix) * mean_view
        updated = self.update(torch.cat([base_feature, fused_view], dim=1))
        return self.norm(base_feature + self.dropout(updated)), view_weights


class MultiScaleLocalViewEncoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.scale_values = [int(value) for value in param.local_multiscale_topk]
        self.scale_encoders = nn.ModuleDict()
        for scale in self.scale_values:
            self.scale_encoders[str(scale)] = LocalGraphTransformerEncoder(
                input_dim=param.outSize,
                num_layers=param.local_encoder_layers,
                dropout=param.local_encoder_dropout,
                num_heads=param.num_heads1,
                hidden_dim=param.local_encoder_hidden_dim,
            )

        self.scale_fusion = MultiViewAttentionFusion(
            input_dim=param.outSize,
            hidden_dim=param.local_encoder_hidden_dim,
            dropout=param.local_encoder_dropout,
            attention_mix=param.local_attention_mix,
        )
    def forward(self, graph_entries, base_feature):
        scale_features = [
            self.scale_encoders[str(scale)](graph_entries[scale], base_feature)
            for scale in self.scale_values
        ]
        fused_feature, scale_weights = self.scale_fusion(base_feature, scale_features)
        return fused_feature, scale_weights


class PairScoringHead(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.layeratt_m = MultiHeadAttention(param.outSize, param.outSize, param.num_heads1)
        self.layeratt_d = MultiHeadAttention(param.outSize, param.outSize, param.num_heads1)
        self.fc_linear = MLP(param.outSize, 1, dropout=param.fcDropout, actFunc=nn.ReLU()).to(param.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pair_data, miRNA_embed, disease_embed):
        miRNA_feature, disease_feature = pro_data(pair_data, miRNA_embed, disease_embed)
        out_m = self.layeratt_m(miRNA_feature)
        out_d = self.layeratt_d(disease_feature)
        pair_embed = out_m * out_d
        logits = self.fc_linear(pair_embed).squeeze(dim=1)
        prob = self.sigmoid(logits)
        return {
            "prob": prob,
            "logits": logits,
            "pair_embed": pair_embed,
        }


class MultiScaleLocalTransformerEncoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.miRNA_view_keys = ["mm_f", "mm_s", "mm_g"]
        self.disease_view_keys = ["dd_t", "dd_s", "dd_g"]
        self.miRNA_view_encoders = nn.ModuleDict({key: MultiScaleLocalViewEncoder(param) for key in self.miRNA_view_keys})
        self.disease_view_encoders = nn.ModuleDict({key: MultiScaleLocalViewEncoder(param) for key in self.disease_view_keys})
        self.miRNA_fusion = MultiViewAttentionFusion(
            input_dim=param.outSize,
            hidden_dim=param.local_encoder_hidden_dim,
            dropout=param.local_encoder_dropout,
            attention_mix=param.local_attention_mix,
        )
        self.disease_fusion = MultiViewAttentionFusion(
            input_dim=param.outSize,
            hidden_dim=param.local_encoder_hidden_dim,
            dropout=param.local_encoder_dropout,
            attention_mix=param.local_attention_mix,
        )

    def forward(self, sim_data, miRNA_embed, disease_embed):
        local_graphs = sim_data["local_graphs"]
        miRNA_views = []
        disease_views = []
        scale_weight_vectors = []

        for key in self.miRNA_view_keys:
            fused_view, scale_weights = self.miRNA_view_encoders[key](local_graphs["miRNA"][key], miRNA_embed)
            miRNA_views.append(fused_view)
            scale_weight_vectors.append(scale_weights.mean(dim=0))

        for key in self.disease_view_keys:
            fused_view, scale_weights = self.disease_view_encoders[key](local_graphs["disease"][key], disease_embed)
            disease_views.append(fused_view)
            scale_weight_vectors.append(scale_weights.mean(dim=0))

        fused_m, miRNA_weights = self.miRNA_fusion(miRNA_embed, miRNA_views)
        fused_d, disease_weights = self.disease_fusion(disease_embed, disease_views)
        local_scale_weight_mean = torch.stack(scale_weight_vectors, dim=0).mean(dim=0)
        return fused_m, fused_d, {
            "local_scale_weight_mean": local_scale_weight_mean.detach(),
            "miRNA_view_weight_mean": miRNA_weights.detach().mean(dim=0),
            "disease_view_weight_mean": disease_weights.detach().mean(dim=0),
        }


class PGLRMDA(nn.Module):
    def __init__(self, param, m_emd, d_emd):
        super(PGLRMDA, self).__init__()
        self.outSize = param.outSize
        self.device = param.device
        self.num_miRNA = param.num_miRNA
        self.num_disease = param.num_disease
        self.local_knn_encoder = MultiScaleLocalTransformerEncoder(param)
        self.local_pair_head = PairScoringHead(param)
        self.global_branch = EnhancedGlobalBranch(param)
        self.args = param
        self.Xm = m_emd
        self.Xd = d_emd
        self.fusion_gate = ReliabilityGateFusion(embed_dim=self.outSize)
        self.latest_forward_stats = {}

        generator = torch.Generator().manual_seed(param.seed)
        self.register_buffer("m_init_feature", torch.randn(self.num_miRNA, self.args.fm, generator=generator))
        self.register_buffer("d_init_feature", torch.randn(self.num_disease, self.args.fd, generator=generator))

    def prepare_batch_invariant_state(self, sim_data):
        em = self.Xm(sim_data, self.m_init_feature)
        ed = self.Xd(sim_data, self.d_init_feature)
        local_m, local_d, local_stats = self.local_knn_encoder(sim_data, em, ed)
        global_state = self.global_branch.encode_graph_state(sim_data, em, ed)
        return {
            "local_m": local_m,
            "local_d": local_d,
            "local_stats": local_stats,
            "global_state": global_state,
        }

    def _predict_from_state(self, state, train_data):
        local_output = self.local_pair_head(train_data, state["local_m"], state["local_d"])
        global_output = self.global_branch.predict_from_graph_state(train_data, state["global_state"])

        gate_output = self.fusion_gate(
            local_output["pair_embed"],
            global_output["pair_embed"],
            local_output["logits"],
            global_output["logits"],
        )
        gate_reg_loss = (
            (1.0 - gate_output["disagreement"]) * torch.square(gate_output["alpha_raw"] - 0.5)
        ).mean() * self.args.gate_reg_weight
        alpha = gate_output["alpha"]

        prediction = alpha * global_output["prob"] + (1.0 - alpha) * local_output["prob"]
        local_stats = state["local_stats"]
        self.latest_forward_stats = {
            "gate_values": alpha.detach(),
            "gate_mean": float(alpha.detach().mean().item()),
            "gate_std": float(alpha.detach().std(unbiased=False).item()),
            "gate_reg_loss": gate_reg_loss,
            "local_reliability_values": gate_output["local_reliability"].detach(),
            "global_reliability_values": gate_output["global_reliability"].detach(),
            "branch_disagreement_values": gate_output["disagreement"].detach(),
            "prob_gap_values": gate_output["prob_gap"].detach(),
            "repr_gap_values": gate_output["repr_gap"].detach(),
            "local_prob_values": gate_output["local_prob"].detach(),
            "global_prob_values": gate_output["global_prob"].detach(),
            "final_prob_values": prediction.detach(),
            "local_reliability_mean": float(gate_output["local_reliability"].detach().mean().item()),
            "global_reliability_mean": float(gate_output["global_reliability"].detach().mean().item()),
            "branch_disagreement_mean": float(gate_output["disagreement"].detach().mean().item()),
            "prob_gap_mean": float(gate_output["prob_gap"].detach().mean().item()),
            "repr_gap_mean": float(gate_output["repr_gap"].detach().mean().item()),
        }
        for projection_stat_name in ["miRNA_projection_weight_mean", "disease_projection_weight_mean"]:
            if projection_stat_name in global_output:
                projection_weights = global_output[projection_stat_name].detach()
                self.latest_forward_stats[f"{projection_stat_name}_tensor"] = projection_weights
                self.latest_forward_stats[projection_stat_name] = projection_weights.cpu().tolist()
        for projection_scalar_name in [
            "miRNA_projection_edge_count",
            "miRNA_projection_density",
            "miRNA_projection_edge_weight_mean",
            "miRNA_projection_edge_weight_max",
            "disease_projection_edge_count",
            "disease_projection_density",
            "disease_projection_edge_weight_mean",
            "disease_projection_edge_weight_max",
        ]:
            if projection_scalar_name in global_output:
                self.latest_forward_stats[projection_scalar_name] = float(global_output[projection_scalar_name])
        if "local_scale_weight_mean" in local_stats:
            self.latest_forward_stats["local_scale_weight_mean_tensor"] = local_stats["local_scale_weight_mean"].detach()
            self.latest_forward_stats["local_scale_weight_mean"] = (
                local_stats["local_scale_weight_mean"].detach().cpu().tolist()
            )
        if "miRNA_view_weight_mean" in local_stats:
            self.latest_forward_stats["miRNA_view_weight_mean_tensor"] = local_stats["miRNA_view_weight_mean"].detach()
            self.latest_forward_stats["miRNA_view_weight_mean"] = (
                local_stats["miRNA_view_weight_mean"].detach().cpu().tolist()
            )
        if "disease_view_weight_mean" in local_stats:
            self.latest_forward_stats["disease_view_weight_mean_tensor"] = local_stats["disease_view_weight_mean"].detach()
            self.latest_forward_stats["disease_view_weight_mean"] = (
                local_stats["disease_view_weight_mean"].detach().cpu().tolist()
            )
        return prediction

    def predict_from_cached_state(self, state, train_data):
        return self._predict_from_state(state, train_data)

    def forward(self, sim_data, train_data):
        state = self.prepare_batch_invariant_state(sim_data)
        return self._predict_from_state(state, train_data)

    def get_latest_forward_stats(self):
        return self.latest_forward_stats


def pro_data(data, em, ed):
    edgeData = data.t()
    m_index = edgeData[0].long()
    d_index = edgeData[1].long()
    Em = torch.index_select(em, 0, m_index)
    Ed = torch.index_select(ed, 0, d_index)
    return Em, Ed


def pair_context_features(miRNA_feature, disease_feature):
    return torch.cat(
        [
            miRNA_feature,
            disease_feature,
            torch.abs(miRNA_feature - disease_feature),
            miRNA_feature * disease_feature,
        ],
        dim=1,
    )


class RelationAwareHeteroEncoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.device = param.device
        self.layers = nn.ModuleList(
            [
                HeteroGraphConv(
                    {
                        "md": GraphConv(param.outSize, param.outSize, norm="both", allow_zero_in_degree=True),
                        "dm": GraphConv(param.outSize, param.outSize, norm="both", allow_zero_in_degree=True),
                    },
                    aggregate="sum",
                )
                for _ in range(param.global_rel_layers)
            ]
        )
        self.dropout = nn.Dropout(param.global_rel_dropout)
        self.miRNA_norms = nn.ModuleList([nn.LayerNorm(param.outSize) for _ in range(param.global_rel_layers)])
        self.disease_norms = nn.ModuleList([nn.LayerNorm(param.outSize) for _ in range(param.global_rel_layers)])

    def forward(self, sdata, em, ed):
        graph = _ensure_hetero_graph_entry_device(sdata["m_d_hetero"], self.device)
        features = {"miRNA": em, "disease": ed}

        for layer, miRNA_norm, disease_norm in zip(self.layers, self.miRNA_norms, self.disease_norms):
            updated = layer(graph, features)
            features = {
                "miRNA": miRNA_norm(features["miRNA"] + self.dropout(torch.relu(updated["miRNA"]))),
                "disease": disease_norm(features["disease"] + self.dropout(torch.relu(updated["disease"]))),
            }

        return features["miRNA"], features["disease"]


class ProjectionGraphEncoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.device = param.device
        self.edge_weight_norm = EdgeWeightNorm(norm="both")
        self.layers = nn.ModuleList(
            [
                GraphConv(
                    param.outSize,
                    param.outSize,
                    norm="none",
                    allow_zero_in_degree=True,
                )
                for _ in range(param.global_rel_layers)
            ]
        )
        self.dropout = nn.Dropout(param.global_rel_dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(param.outSize) for _ in range(param.global_rel_layers)])

    def forward(self, graph_entry, features):
        graph, edge_weight = _ensure_graph_entry_device(graph_entry, self.device)
        if graph.num_edges() == 0:
            return features

        norm_weight = self.edge_weight_norm(graph, edge_weight)
        hidden = features
        for layer, norm in zip(self.layers, self.norms):
            updated = torch.relu(layer(graph, hidden, edge_weight=norm_weight))
            hidden = norm(hidden + self.dropout(updated))
        return hidden


class BipartiteProjectionEnhancedEncoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        hidden_dim = getattr(param, "global_projection_hidden_dim", 128)
        self.relation_encoder = RelationAwareHeteroEncoder(param)
        self.miRNA_projection_encoder = ProjectionGraphEncoder(param)
        self.disease_projection_encoder = ProjectionGraphEncoder(param)
        self.miRNA_fusion = MultiViewAttentionFusion(
            input_dim=param.outSize,
            hidden_dim=hidden_dim,
            dropout=param.global_rel_dropout,
            attention_mix=1.0,
        )
        self.disease_fusion = MultiViewAttentionFusion(
            input_dim=param.outSize,
            hidden_dim=hidden_dim,
            dropout=param.global_rel_dropout,
            attention_mix=1.0,
        )

    def _projection_stats(self, projection_graphs):
        source_stats = projection_graphs.get("stats", {})
        stats = {}
        for node_type in ["miRNA", "disease"]:
            prefix = f"{node_type}_projection"
            for suffix in ["edge_count", "density", "edge_weight_mean", "edge_weight_max"]:
                key = f"{prefix}_{suffix}"
                stats[key] = float(source_stats.get(key, 0.0))
        return stats

    def forward(self, sdata, em, ed):
        hetero_miRNA, hetero_disease = self.relation_encoder(sdata, em, ed)
        projection_graphs = sdata["bipartite_projection_graphs"]
        projected_miRNA = self.miRNA_projection_encoder(projection_graphs["miRNA"], em)
        projected_disease = self.disease_projection_encoder(projection_graphs["disease"], ed)
        fused_miRNA, miRNA_projection_weights = self.miRNA_fusion(
            hetero_miRNA,
            [hetero_miRNA, projected_miRNA],
        )
        fused_disease, disease_projection_weights = self.disease_fusion(
            hetero_disease,
            [hetero_disease, projected_disease],
        )
        stats = {
            "miRNA_projection_weight_mean": miRNA_projection_weights.detach().mean(dim=0),
            "disease_projection_weight_mean": disease_projection_weights.detach().mean(dim=0),
        }
        stats.update(self._projection_stats(projection_graphs))
        return fused_miRNA, fused_disease, stats


class PairContextEncoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        hidden_dim = getattr(param, "global_projection_hidden_dim", 128)
        self.project = nn.Sequential(
            nn.Linear(param.outSize * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(param.global_rel_dropout),
            nn.Linear(hidden_dim, param.outSize),
            nn.LayerNorm(param.outSize),
        )

    def forward(self, miRNA_feature, disease_feature):
        context = pair_context_features(miRNA_feature, disease_feature)
        return self.project(context), context


class GlobalPredictionHead(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.fc_linear = MLP(param.outSize, 1, dropout=param.fcDropout, actFunc=nn.ReLU()).to(param.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pair_embed):
        logits = self.fc_linear(pair_embed).squeeze(dim=1)
        prob = self.sigmoid(logits)
        return {
            "prob": prob,
            "logits": logits,
            "pair_embed": pair_embed,
        }


class EnhancedGlobalBranch(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.relational_encoder = BipartiteProjectionEnhancedEncoder(param)
        self.pair_context_encoder = PairContextEncoder(param)
        self.prediction_head = GlobalPredictionHead(param)

    def encode_graph_state(self, sdata, em, ed):
        miRNA_nodes, disease_nodes, projection_stats = self.relational_encoder(sdata, em, ed)
        return {
            "miRNA_nodes": miRNA_nodes,
            "disease_nodes": disease_nodes,
            "projection_stats": projection_stats,
        }

    def predict_from_graph_state(self, tdata, graph_state):
        miRNA_pair, disease_pair = pro_data(tdata, graph_state["miRNA_nodes"], graph_state["disease_nodes"])
        global_pair_embed, _ = self.pair_context_encoder(miRNA_pair, disease_pair)

        output = self.prediction_head(global_pair_embed)
        output["hetero_pair_embed"] = global_pair_embed
        output.update(graph_state["projection_stats"])
        return output

    def forward(self, sdata, tdata, em, ed):
        graph_state = self.encode_graph_state(sdata, em, ed)
        return self.predict_from_graph_state(tdata, graph_state)


class EmbeddingM(nn.Module):
    def __init__(self, args):
        super(EmbeddingM, self).__init__()
        self.args = args
        self.edge_weight_norm = EdgeWeightNorm(norm="both")
        self.gcn_x1_f = GraphConv(self.args.fm, self.args.fm, norm="none", allow_zero_in_degree=True)
        self.gcn_x2_f = GraphConv(self.args.fm, self.args.fm, norm="none", allow_zero_in_degree=True)
        self.gcn_x1_s = GraphConv(self.args.fm, self.args.fm, norm="none", allow_zero_in_degree=True)
        self.gcn_x2_s = GraphConv(self.args.fm, self.args.fm, norm="none", allow_zero_in_degree=True)
        self.gcn_x1_g = GraphConv(self.args.fm, self.args.fm, norm="none", allow_zero_in_degree=True)
        self.gcn_x2_g = GraphConv(self.args.fm, self.args.fm, norm="none", allow_zero_in_degree=True)

        self.fc1_x = nn.Linear(self.args.view * self.args.gcn_layers1,
                               5 * self.args.view * self.args.gcn_layers1)
        self.fc2_x = nn.Linear(5 * self.args.view * self.args.gcn_layers1,
                               self.args.view * self.args.gcn_layers1)
        self.sigmoidx = nn.Sigmoid()
        self.cnn_x = nn.Conv2d(self.args.view * self.args.gcn_layers1, 1, kernel_size=(1, 1), stride=1, bias=True)

    def _run_branch(self, conv1, conv2, graph_entry, features):
        graph, edge_weight = _ensure_graph_entry_device(graph_entry, features.device)
        norm_weight = self.edge_weight_norm(graph, edge_weight)
        out1 = torch.relu(conv1(graph, features, edge_weight=norm_weight))
        out2 = torch.relu(conv2(graph, out1, edge_weight=norm_weight))
        return out1, out2

    def forward(self, data, fm1):
        x_f1, x_f2 = self._run_branch(self.gcn_x1_f, self.gcn_x2_f, data["mm_f"], fm1)
        x_s1, x_s2 = self._run_branch(self.gcn_x1_s, self.gcn_x2_s, data["mm_s"], fm1)
        x_g1, x_g2 = self._run_branch(self.gcn_x1_g, self.gcn_x2_g, data["mm_g"], fm1)

        XM = torch.cat((x_f1, x_f2, x_s1, x_s2, x_g1, x_g2), dim=1).t()
        XM = XM.view(1, self.args.view * self.args.gcn_layers1, self.args.fm, -1)
        x_channel_attention = XM.mean(dim=(2, 3))
        x_channel_attention = torch.relu(self.fc1_x(x_channel_attention))
        x_channel_attention = self.sigmoidx(self.fc2_x(x_channel_attention))
        x_channel_attention = x_channel_attention.view(1, -1, 1, 1)
        XM_channel_attention = torch.relu(x_channel_attention * XM)
        return self.cnn_x(XM_channel_attention).squeeze(0).squeeze(0).t()


class EmbeddingD(nn.Module):
    def __init__(self, args):
        super(EmbeddingD, self).__init__()
        self.args = args
        self.edge_weight_norm = EdgeWeightNorm(norm="both")
        self.gcn_y1_t = GraphConv(self.args.fd, self.args.fd, norm="none", allow_zero_in_degree=True)
        self.gcn_y2_t = GraphConv(self.args.fd, self.args.fd, norm="none", allow_zero_in_degree=True)
        self.gcn_y1_s = GraphConv(self.args.fd, self.args.fd, norm="none", allow_zero_in_degree=True)
        self.gcn_y2_s = GraphConv(self.args.fd, self.args.fd, norm="none", allow_zero_in_degree=True)
        self.gcn_y1_g = GraphConv(self.args.fd, self.args.fd, norm="none", allow_zero_in_degree=True)
        self.gcn_y2_g = GraphConv(self.args.fd, self.args.fd, norm="none", allow_zero_in_degree=True)

        self.fc1_y = nn.Linear(self.args.view * self.args.gcn_layers1,
                               5 * self.args.view * self.args.gcn_layers1)
        self.fc2_y = nn.Linear(5 * self.args.view * self.args.gcn_layers1,
                               self.args.view * self.args.gcn_layers1)
        self.sigmoidy = nn.Sigmoid()
        self.cnn_y = nn.Conv2d(self.args.view * self.args.gcn_layers1, 1, kernel_size=(1, 1), stride=1, bias=True)

    def _run_branch(self, conv1, conv2, graph_entry, features):
        graph, edge_weight = _ensure_graph_entry_device(graph_entry, features.device)
        norm_weight = self.edge_weight_norm(graph, edge_weight)
        out1 = torch.relu(conv1(graph, features, edge_weight=norm_weight))
        out2 = torch.relu(conv2(graph, out1, edge_weight=norm_weight))
        return out1, out2

    def forward(self, data, dm1):
        y_d_t1, y_d_t2 = self._run_branch(self.gcn_y1_t, self.gcn_y2_t, data["dd_t"], dm1)
        y_d_s1, y_d_s2 = self._run_branch(self.gcn_y1_s, self.gcn_y2_s, data["dd_s"], dm1)
        y_d_g1, y_d_g2 = self._run_branch(self.gcn_y1_g, self.gcn_y2_g, data["dd_g"], dm1)

        combined_embedding = torch.cat((y_d_t1, y_d_t2, y_d_s1, y_d_s2, y_d_g1, y_d_g2), 1).t()
        combined_embedding = combined_embedding.view(1, self.args.view * self.args.gcn_layers1, self.args.fd, -1)
        y_channel_attention = combined_embedding.mean(dim=(2, 3))
        y_channel_attention = torch.relu(self.fc1_y(y_channel_attention))
        y_channel_attention = self.sigmoidy(self.fc2_y(y_channel_attention))
        y_channel_attention = y_channel_attention.view(1, -1, 1, 1)
        YD_channel_attention = torch.relu(y_channel_attention * combined_embedding)
        return self.cnn_y(YD_channel_attention).squeeze(0).squeeze(0).t()


class MultiHeadAttention(nn.Module):
    def __init__(self, inSize, outSize, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([LayerAtt2(inSize, outSize) for _ in range(num_heads)])
        self.merge_layer = nn.Linear(num_heads * outSize, outSize)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        return self.merge_layer(torch.cat(head_outputs, dim=-1))
