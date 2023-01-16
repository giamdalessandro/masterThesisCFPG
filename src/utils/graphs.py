import torch


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
	""" ZAVVE: Creates the dense version of adjacency matrix `graph`, considering
	only those edges also present in `mask`.

	Args
	- `graph`  : graph sparse adjacency matrix;
	- `mask`   : masked sparse adjacency matrix;
	- `n_rows` : rows of the dense adjacency matrix, i.e. number of nodes in the graph;
	"""
	rows = graph[0]
	cols = graph[1]

	dense_mat = torch.zeros((n_rows,n_rows))
	dense_mat[rows,cols] = mask

	return dense_mat

def create_symm_matrix_from_vec(vector, n_rows):
	matrix = torch.zeros(n_rows, n_rows)
	idx = torch.tril_indices(n_rows, n_rows)
	matrix[idx[0], idx[1]] = vector
	symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
	return symm_matrix

def create_vec_from_symm_matrix(matrix, P_vec_size):
	idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
	vector = matrix[idx[0], idx[1]]
	return vector