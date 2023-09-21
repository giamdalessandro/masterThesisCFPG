import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph


def index_edge(graph, pair):
    return torch.where((graph.T == pair).all(dim=1))[0]

def normalize_adj(adj):
	"""Normalize adjacancy matrix according to reparam trick in GCN paper"""
	A_tilde = adj + torch.eye(adj.shape[0])
	D_tilde = torch.diag(sum(A_tilde))
	# Raise to power -1/2, set all infs to 0s
	D_tilde_exp = D_tilde ** (-1 / 2)
	D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

	# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
	norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
	return norm_adj

def sparse_to_dense_adj(graph, mask, n_rows: int):
	"""ZAVVE: Creates the dense version of adjacency matrix `graph`, considering
	only those edges also present in `mask`.

	#### Args
	graph : `torch.Tensor`
		graph sparse adjacency matrix;
	mask : `torch.Tensor` 
		masked sparse adjacency matrix;
	n_rows : `int` 
		rows of the dense adjacency matrix, i.e. number of nodes in the graph;
	"""
	rows = graph[0]
	cols = graph[1]

	dense_mat = torch.zeros((n_rows,n_rows))
	dense_mat[rows,cols] = mask

	return dense_mat

def create_symm_matrix_from_vec(vector, n_rows):
	"""Create a symmetric matrix with the elements of `vector` symmetric to the main diagonal."""
	matrix = torch.zeros(n_rows, n_rows)
	idx = torch.tril_indices(n_rows, n_rows)
	matrix[idx[0], idx[1]] = vector
	symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
	return symm_matrix

def create_vec_from_symm_matrix(matrix, P_vec_size):
	"""Create a vector with the elements of symmetric matrix `matrix`."""
	idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
	vector = matrix[idx[0], idx[1]]
	return vector



def get_degree_matrix(adj):
	return torch.diag(sum(adj))

def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
	edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])                      # Get all nodes involved
	edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)  # Get relabelled subset of edges
	sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
	sub_feat = features[edge_subset[0], :]
	sub_labels = labels[edge_subset[0]]
	new_index = np.array([i for i in range(len(edge_subset[0]))])
	node_dict = dict(zip(edge_subset[0].numpy(), new_index))                           # Maps orig labels to new
	# print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
	return sub_adj, sub_feat, sub_labels, node_dict

