import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, dropout, log_softmax, nll_loss
from torch.nn.parameter import Parameter
#from torch_geometric.nn import GCNConv


## from gcn.py
class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

## from gcn.py
class GCNSynthetic(nn.Module):
    """3-layer GCN used in GNN Explainer synthetic tasks"""
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCNSynthetic, self).__init__()
        self.nclass = nclass

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        input_lin = self.embedding(x, adj)
        x = self.lin(input_lin)
        return log_softmax(x, dim=1)

    def embedding(self, x, adj):
        """
        Computes nodes embeddings, i.e. feed-forward only through the
        convolutional layers of the GCN.
        """
        x1 = relu(self.gc1(x, adj))
        x1 = dropout(x1, self.dropout, training=self.training)
        x2 = relu(self.gc2(x1, adj))
        x2 = dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, adj)
        input_lin = torch.cat((x1, x2, x3), dim=1)

        return input_lin

    def loss(self, pred, label):
        return nll_loss(pred, label)
    


## from gcn_perturb.py
import torch.nn.functional as F
from utils.graphs import create_symm_matrix_from_vec, create_vec_from_symm_matrix

## from gcn_perturb.py
class GraphConvolutionPerturb(nn.Module):
	"""Similar to GraphConvolution except includes P_hat"""
	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolutionPerturb, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias is not None:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)

	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
		       + str(self.in_features) + ' -> ' \
		       + str(self.out_features) + ')'

## from gcn_perturb.py
class GCNSyntheticPerturb(nn.Module):
	"""3-layer GCN used in GNN Explainer synthetic tasks"""
	def __init__(self, 
			nfeat: int, 
			nhid: int, 
			nout: int, 
			nclass: int, 
			adj: torch.Tensor, 
			dropout: float, 
			beta: float, 
			edge_additions: bool=False,
			device: str="cpu"
		):
		super(GCNSyntheticPerturb, self).__init__()
		self.adj       = adj
		self.nclass    = nclass
		self.beta      = beta
		self.num_nodes = self.adj.shape[0]
		self.edge_additions = edge_additions    # are edge additions included in perturbed matrix
		self.device = device

		# P_hat needs to be symmetric ==> learn vector representing entries in 
		# upper/lower triangular matrix and use to populate P_hat later
		self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes

		if self.edge_additions:
			self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size))).to(self.device)
		else:
			self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size))).to(self.device)

		self.reset_parameters()
    
		self.gc1     = GraphConvolutionPerturb(nfeat, nhid)
		self.gc2     = GraphConvolutionPerturb(nhid, nhid)
		self.gc3     = GraphConvolution(nhid, nout)
		self.lin     = nn.Linear(nhid + nhid + nout, nclass)
		self.dropout = dropout

	def reset_parameters(self, eps: float=10**-4):
		"""New version of reset_parameters for CUDA processing."""
		with torch.no_grad():
			if self.edge_additions:
				adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size) #.numpy()
				first = adj_vec[0] - eps
				adj_vec = torch.add(adj_vec,eps)
				adj_vec[0] = first

				torch.add(self.P_vec, adj_vec)       #self.P_vec is all 0s
			else:
				torch.sub(self.P_vec, eps)

	def _old_reset_parameters(self, eps: float=10**-4):
		# Think more about how to initialize this
		with torch.no_grad():
			if self.edge_additions:
				adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
				for i in range(len(adj_vec)):
					if i < 1:
						adj_vec[i] = adj_vec[i] - eps
					else:
						adj_vec[i] = adj_vec[i] + eps
				torch.add(self.P_vec, torch.FloatTensor(adj_vec))       #self.P_vec is all 0s
			else:
				torch.sub(self.P_vec, eps)

	def forward(self, x, sub_adj, embedding: bool=False):
		self.sub_adj = sub_adj
		# Same as normalize_adj in utils.py except includes P_hat in A_tilde
		self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)  # Ensure symmetry

		A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
		A_tilde.requires_grad = True

		if self.edge_additions:  # Learn new adj matrix directly
			# Use sigmoid to bound P_hat in [0,1]
			A_tilde = torch.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes)  
		else:  
			# Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
			# Use sigmoid to bound P_hat in [0,1]
			A_tilde = torch.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes)       

		D_tilde = torch.diag(sum(A_tilde)).detach()   # Don't need gradient of this
		# Raise to power -1/2, set all infs to 0s
		D_tilde_exp = D_tilde.pow(-0.5)               # D_tilde ** (-1 / 2)
		D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
		norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

		x1 = F.relu(self.gc1(x, norm_adj))
		x1 = F.dropout(x1, self.dropout, training=self.training)
		x2 = F.relu(self.gc2(x1, norm_adj))
		x2 = F.dropout(x2, self.dropout, training=self.training)
		x3 = self.gc3(x2, norm_adj)
		if embedding: return torch.cat((x1, x2, x3), dim=1)
			
		x = self.lin(torch.cat((x1, x2, x3), dim=1))
		return F.log_softmax(x, dim=1)

	def forward_prediction(self, x, P_mask=None):
		"""Same as forward but uses P instead of P_hat ==> non-differentiable,
		but needed for the actual predictions.
		"""
		if P_mask is None:
			self.P = (torch.sigmoid(self.P_hat_symm) >= 0.5).float()   # threshold P_hat
		else:
			self.P = (P_mask.sigmoid() >= 0.5).float()
		self.P = self.P.to(self.device)


		if self.edge_additions:
			A_tilde = self.P + torch.eye(self.num_nodes).to(self.device)  # ZAVVE cuda test
		else:
			A_tilde = self.P * self.adj + torch.eye(self.num_nodes).to(self.device)  # ZAVVE cuda test

		D_tilde = torch.diag(sum(A_tilde))
		# Raise to power -1/2, set all infs to 0s
		D_tilde_exp = D_tilde.pow(-0.5)     #D_tilde ** (-1 / 2)
		D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
		norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

		x1 = F.relu(self.gc1(x, norm_adj))
		x1 = F.dropout(x1, self.dropout, training=self.training)
		x2 = F.relu(self.gc2(x1, norm_adj))
		x2 = F.dropout(x2, self.dropout, training=self.training)
		x3 = self.gc3(x2, norm_adj)
		input_lin = torch.cat((x1, x2, x3), dim=1)
		x = self.lin(input_lin)
		
		return F.log_softmax(x, dim=1), self.P, x3

	def loss(self, output, y_pred_orig, y_pred_new_actual):
		pred_same = (y_pred_new_actual == y_pred_orig).float()

		# Need dim >=2 for F.nll_loss to work
		output = output.unsqueeze(0)
		y_pred_orig = y_pred_orig.unsqueeze(0)

		if self.edge_additions:
			cf_adj = self.P
		else:
			cf_adj = self.P * self.adj
		cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

		# Want negative in front to maximize loss instead of minimizing it to find CFs
		loss_pred = -F.nll_loss(output, y_pred_orig)              #F.cross_entropy(output, y_pred_orig)
		loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) / 2    # num of edges changed (symmetrical)

		# Zero-out loss_pred with pred_same if prediction flips
		loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
		return loss_total, loss_pred, loss_graph_dist, cf_adj
