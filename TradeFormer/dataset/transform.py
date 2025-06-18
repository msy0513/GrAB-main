import networkx as nx
import numpy as np
from functools import partial
from typing import Any, Optional
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


class RRWPTransform(object):
    def __init__(self, ksteps=8):
        """Initializing positional encoding with RRWP"""
        self.transform = partial(add_full_rrwp, walk_length=ksteps)

    def __call__(self, data):
        data = self.transform(data)
        return data


def add_node_attr(data: Data, value: Any, attr_name: Optional[str] = None) -> Data:
    data[attr_name] = value
    return data


@torch.no_grad()
def add_full_rrwp(data, walk_length=8, attr_name_abs="rrwp"):
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    print(adj)
    adj = adj.to_dense()

    pe_list = [torch.eye(num_nodes, dtype=torch.float)]
    pe_list.append(adj)

    if walk_length <= 2:
        raise ValueError("walk_length must be greater than 2")

    out = adj
    for j in range(len(pe_list), walk_length):
        print("walk", j)
        out = out @ adj
        pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)  # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_abs}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_abs}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data


@torch.no_grad()
def add_similarity_rrwp(
    data, walk_length=8, attr_name_abs="similar_rrwp", add_dissimilarity=True
):
    if walk_length <= 2:
        raise ValueError("walk_length must be greater than 2")

    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()
    dissimilarity_adj = adj.clone()

    for source in range(num_nodes):
        target_index = torch.nonzero(adj[source])
        source_feature = data.x[source]
        score_list = []
        dissimilarity_list = []

        for tmp in target_index:
            tmp_feature = data.x[tmp]
            cos_sim = F.cosine_similarity(
                source_feature.float(), tmp_feature.float(), dim=-1
            )
            score_list.append(cos_sim)
            dissimilarity_list.append((-1) * cos_sim)

        score_list = torch.tensor(score_list)
        weight = torch.softmax(score_list, 0)
        for element, new_weight in zip(target_index, weight):
            adj[source][element] = new_weight

        dissimilarity_list = torch.tensor(dissimilarity_list)
        weight_dissimilarity = torch.softmax(dissimilarity_list, 0)
        for element, new_weight in zip(target_index, weight_dissimilarity):
            dissimilarity_adj[source][element] = new_weight

    if add_dissimilarity:
        p_list = [torch.eye(num_nodes, dtype=torch.float)]
        p_list.append(dissimilarity_adj)

        out = dissimilarity_adj
        for j in range(len(p_list), walk_length):
            out = out @ dissimilarity_adj
            p_list.append(out)

        p = torch.stack(p_list, dim=-1)  # n x n x k
        abs_p = p.diagonal().transpose(0, 1)  # n x k
        rel_p = SparseTensor.from_dense(p, has_value=True)
        rel_p_row, rel_p_col, rel_p_val = rel_p.coo()
        rel_p_idx = torch.stack([rel_p_row, rel_p_col], dim=0)

        data = add_node_attr(data, abs_p, attr_name="dissimilar_rrwp")
        data = add_node_attr(data, rel_p_idx, attr_name="dissimilar_rrwp_index")
        data = add_node_attr(data, rel_p_val, attr_name="dissimilar_rrwp_val")

    pe_list = [torch.eye(num_nodes, dtype=torch.float)]
    pe_list.append(adj)

    out = adj
    for j in range(len(pe_list), walk_length):
        out = out @ adj
        pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)  # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_abs}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_abs}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    custom_one_hot_label = custom_one_hot(data.y, 2)
    data = add_node_attr(data, custom_one_hot_label, attr_name="binary_label")

    addition_feature = get_addtional_feature(data)
    data = add_node_attr(data, addition_feature, attr_name="addition")
    return data


def get_addtional_feature(data):
    G = to_networkx(data, to_undirected=False)
    num_nodes = data.num_nodes
    pagerank_values = nx.pagerank(G)

    clustering_coefficients = nx.clustering(G)
    clustering_tensor = torch.tensor(list(dict(clustering_coefficients).values()))
    clustering_tensor = torch.unsqueeze(clustering_tensor, dim=1)

    betweenness_centrality = nx.betweenness_centrality(G)
    betweenness_tensor = torch.tensor(list(dict(betweenness_centrality).values()))
    betweenness_tensor = torch.unsqueeze(betweenness_tensor, dim=1)

    closeness_centrality = nx.closeness_centrality(G)
    closeness_tensor = torch.tensor(list(dict(closeness_centrality).values()))
    closeness_tensor = torch.unsqueeze(closeness_tensor, dim=1)

    pagerank_coes = {key: value * num_nodes for key, value in pagerank_values.items()}
    pagerank_tensor = torch.tensor(list(pagerank_coes.values()))
    pagerank_tensor = torch.unsqueeze(pagerank_tensor, dim=1)

    merged_tensor = torch.cat(
        (clustering_tensor, pagerank_tensor, betweenness_tensor, closeness_tensor),
        dim=1,
    )

    return merged_tensor


def custom_one_hot(labels, num_classes):
    one_hot_labels = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        if label == 0:
            one_hot_labels[i][0] = 1.0
        elif label == 1:
            one_hot_labels[i][1] = 1.0
    return torch.from_numpy(one_hot_labels)


class SimilarTransform(object):
    """Initializing positional encoding with RRWP and other addtional_feature"""

    def __init__(self, ksteps=8):
        self.transform = partial(add_similarity_rrwp, walk_length=ksteps)
        self.transform_origin = partial(add_full_rrwp, walk_length=ksteps)

    def __call__(self, data):
        data = self.transform(data)
        data = self.transform_origin(data)
        return data
