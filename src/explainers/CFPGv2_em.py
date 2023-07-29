import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

#from utils.graphs import create_symm_matrix_from_vec
from .sparsemax import Sparsemax


def _get_edge_repr(sub_index, enc_embeds, node_id, device: str="cpu"):
    """Use encoder node embeddings to create encoder edge embeddings,
    getting each edge embed by concatenating the embeddings of the nodes 
    adjacent to it with the embeddig of `node_id`, the node which we 
    wish to explain.   

    ### Args
    sub_index : `torch.Tensor`
        edge_index of the neighborhood subgraph of `node_id`;

    enc_embeds : `torch.Tensor`
        embedding of all nodes in `node_id` neighborhood
    
    node_id : int
        id of the node we wish to explain
    
    ### Returns
        Concatenated encoder edge embeddings
    """
    rows = sub_index[0]
    cols = sub_index[1]
    row_embeds = enc_embeds[rows]
    col_embeds = enc_embeds[cols]

    node_embed = enc_embeds[node_id].repeat(rows.size(0), 1).to(device)
    parsed_rep = torch.cat([row_embeds, col_embeds, node_embed], 1).to(device)
    
    return parsed_rep

def _sample_graph(sampling_weights, temperature=1.0, bias=0.0, training=True, device: str="cpu"):
    r"""Implementation of the reparamerization trick to obtain a sample 
    graph while maintaining the posibility to backprop.
    
    ### Args
    sampling_weights : `torch.Tensor`
        Weights provided by the mlp;

    temperature : `float`
        annealing temperature to make the procedure more deterministic;

    bias : `float`
        Bias on the weights to make samplign less deterministic;

    training : `bool`
        If set to false, the samplign will be entirely deterministic;
    
    ### Return 
        sampled graph.
    """
    if training:
        bias = bias + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
        gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(device)
        gate_inputs = (gate_inputs + sampling_weights) / temperature
        graph = torch.sigmoid(gate_inputs)
    else:
        graph = torch.sigmoid(sampling_weights)
        #graph = torch.special.logit(graph, eps=1e-8) 
        #graph = torch.special.entr(sampling_weights)
        #graph[torch.isinf(graph)] = 0  # zeros out infinite elements

    return graph



class GCNExplModule(torch.nn.Module):
    """Class for the GCN-conv explanation module of CFPG-v.2"""
    def __init__(self, 
            in_feats: int, 
            enc_hidden: int=20,
            dec_hidden: int=64,
            dropout: float=0.2, 
            device: str="cpu"
        ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.enc_h    = enc_hidden
        self.dec_h    = dec_hidden
        self.device   = device
        self.dropout  = dropout
        self.logs_d = {
            "pre-sample" : [],
            "post-gcn" : [],
        }

        self.enc_gc1 = GCNConv(self.in_feats, self.enc_h)
        self.enc_gc2 = GCNConv(self.enc_h, self.enc_h)

        self.latent_dim = self.enc_h*3
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim*2, self.dec_h),
            torch.nn.ReLU(), #LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(self.dec_h, 1),
            #torch.nn.Softmax(dim=1)  # ZAVVE: testing
        ).to(self.device)
        self.sparsemax = Sparsemax(dim=0).to(self.device)

    def forward(self, x, edge_index, node_id, temp: float=1.0, bias: float=0.0, train: bool=True):
        """Forward step with a GCN encoder."""
        # encoder step
        x1 = F.relu(self.enc_gc1(x, edge_index))
        x1 = F.dropout(x1,self.dropout)
        x2 = F.relu(self.enc_gc2(x1, edge_index))
        x2 = F.dropout(x2,self.dropout)
        out_enc = torch.cat((x1,x2),dim=1)
        #out_enc = nn.functional.dropout(out_enc,self.dropout)
        # get edge representation
        z = _get_edge_repr(edge_index, out_enc, node_id)
        
        # decoder step
        out_dec = self.decoder(z)
        if not train:
            self.logs_d["pre-sample"].append(out_dec.detach())
            self.logs_d["post-gcn"].append(torch.reshape(z, (-1,)).detach())
        #self.out_decoder.append(out_dec)
        
        #sampled_mask = _sample_graph(out_dec, temperature=temp, bias=bias, training=train)
        #sampled_mask = F.gumbel_softmax(out_dec, tau=temp, hard=False, dim=0)
        sampled_mask = self.sparsemax(out_dec)

        return sampled_mask


class GATExplModule(torch.nn.Module):
    """Class for the GAT-conv explanation module of CFPG-v.2"""
    def __init__(self, 
            in_feats: int, 
            enc_hidden: int=20,
            dec_hidden: int=64,
            heads: int=1,
            add_att: float=0.0,
            dropout: float=0.1, 
            device: str="cpu"
        ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.enc_h    = enc_hidden
        self.dec_h    = dec_hidden
        self.heads    = heads
        self.add_att  = add_att
        self.device   = device
        self.dropout = dropout
        self.logs_d = {
            "pre-sample" : [],
            "post-gcn" : [],
        }

        if self.add_att != 0.0:
            self.enc_gc1 = GATv2Conv(self.in_feats, self.enc_h, self.heads, concat=False, add_self_loops=False)
            self.enc_gc2 = GATv2Conv(self.enc_h, self.enc_h, self.heads, concat=False, add_self_loops=False)
            self.enc_gc3 = GATv2Conv(self.enc_h, self.enc_h, self.heads, concat=False, add_self_loops=False)
        else:
            self.enc_gc1 = GATv2Conv(self.in_feats, self.enc_h, self.heads, concat=False)
            self.enc_gc2 = GATv2Conv(self.enc_h, self.enc_h, self.heads, concat=False)
            self.enc_gc3 = GATv2Conv(self.enc_h, self.enc_h, self.heads, concat=False)

        self.latent_dim = ((self.enc_h*3)*3 + 3) if self.add_att != 0.0 else ((self.enc_h*3)*3) 
        #self.latent_dim = ((self.enc_h*3)*3) 
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.dec_h),
            torch.nn.ReLU(), #LeakyReLU(negative_slope=0.05),
            torch.nn.Linear(self.dec_h, 1),
            #torch.nn.Softmax(dim=1)
        ).to(self.device)

        self.sparsemax = Sparsemax(dim=0).to(self.device)

    def forward(self, x, edge_index, node_id, temp: float=1.0, bias: float=0.0, train: bool=True):
        """Forward step with a GAT encoder."""
        # encoder step
        x1, att_w1 = self.enc_gc1(x, edge_index, return_attention_weights=True)
        x1 = F.dropout(F.relu(x1),self.dropout)
        x2, att_w2 = self.enc_gc2(x1, edge_index, return_attention_weights=True)
        x2 = F.dropout(F.relu(x2),self.dropout)
        x3, att_w3 = self.enc_gc3(x2, edge_index, return_attention_weights=True)
        x3 = F.dropout(F.relu(x3),self.dropout)
        out_enc = torch.cat((x1,x2,x3),dim=1)
        #out_enc = nn.functional.dropout(out_enc,self.dropout)
        # get edge representation
        z = _get_edge_repr(edge_index, out_enc, node_id)

        if self.add_att != 0.0:
            # att_w contains
            att_w1 = torch.mean(att_w1[1], dim=1)#.sigmoid()
            att_w2 = torch.mean(att_w2[1], dim=1)#.sigmoid()
            att_w3 = torch.mean(att_w3[1], dim=1)#.sigmoid()
            #print("\t>> z size:", z.size())
            z = torch.cat((z,att_w1.unsqueeze(dim=1),att_w2.unsqueeze(dim=1),att_w3.unsqueeze(dim=1)), dim=1)

        # decoder step
        out_dec = self.decoder(z)

        if not train:
            self.logs_d["pre-sample"].append(out_dec.detach())
            self.logs_d["post-gcn"].append(torch.reshape(z, (-1,)).detach())

        # add attention
        #if self.add_att != 0.0:
        #    # att_w contains
        #    att_w1 = torch.mean(att_w1[1], dim=1)#.sigmoid()
        #    att_w2 = torch.mean(att_w2[1], dim=1)#.sigmoid()
        #    att_w3 = torch.mean(att_w3[1], dim=1)#.sigmoid()
        #    att_w = (att_w1 + att_w2 + att_w3)#.sigmoid()
        #    out_dec = torch.add(out_dec.squeeze(), att_w, alpha=self.add_att)

        #sampled_mask = _sample_graph(out_dec, temperature=temp, bias=bias, training=train)
        #sampled_mask = F.gumbel_softmax(out_dec, tau=temp, hard=False, dim=0)
        sampled_mask = self.sparsemax(out_dec)
        
        return sampled_mask



class GCNPerturbExplModule(torch.nn.Module):
    """Class for the GCNPerturb-conv explanation module of CFPG-v.2"""
    def __init__(self, 
            in_feats: int, 
            enc_hidden: int=20,
            dec_hidden: int=64,
            dropout: float=0.2,
            edges: torch.Tensor=None,
            num_nodes: int=0,
            edge_addition: bool=True,
            device: str="cpu"
        ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.enc_h    = enc_hidden
        self.dec_h    = dec_hidden
        self.device   = device
        self.dropout  = dropout
        self.num_nodes = num_nodes          # perturbGCN param
        self.num_edges = edges.size(1)      # perturbGCN param
        self.full_adj  = edges              # perturbGCN param
        self.edge_addition = edge_addition  # perturbGCN param

        P_vec_size = self.num_edges
        self.P_vec = torch.nn.Parameter(torch.FloatTensor(torch.ones(P_vec_size))).to(self.device)
        self.enc_gc1 = GCNConv(self.in_feats, self.enc_h)

        self.latent_dim = self.enc_h*3
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.dec_h),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dec_h, 1)
        ).to(self.device)

    def forward(self, x, edge_index, node_id, bias: float=0.0, train: bool=True):
        """Forward step with a GCNSyntheticPerturb encoder."""
        # pre-encoder step
		# Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = self._create_symm_matrix_from_vec(self.P_vec, self.num_nodes)  # Ensure symmetry

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        #A_tilde = torch.FloatTensor(self.P_hat_symm.size())
        A_tilde.requires_grad = True

        if self.edge_addition:  # Learn new adj matrix directly
            # Use sigmoid to bound P_hat in [0,1]
            A_tilde = torch.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes)  
        else:  
            # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            # Use sigmoid to bound P_hat in [0,1]
            A_tilde = torch.sigmoid(self.P_hat_symm) * edge_index + torch.eye(self.num_nodes)       

        D_tilde = torch.diag(A_tilde).detach()     # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde.pow(-0.5)            # D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0  # zeros out infinite elements

		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_edge_index = torch.mul(torch.mul(D_tilde_exp, A_tilde), D_tilde_exp)
        #norm_edge_index = (D_tilde * A_tilde) * D_tilde_exp
        # get values only for edges in edge_index (i.e. the current subgraph)
        norm_edge_index = norm_edge_index[edge_index[0],edge_index[1]]

        # encoder step
        x1 = self.enc_gc1(x, edge_index, norm_edge_index)
        out_enc = F.relu(x1)
        # get edge representation
        z = _get_edge_repr(edge_index, out_enc, node_id)
        # decoder step
        out_dec = self.decoder(z)
        sampled_mask = _sample_graph(out_dec, bias=bias, training=train)

        return sampled_mask

    def _create_symm_matrix_from_vec(self, vector, n_rows):
        """Create symmetric adjacency matrix, considering only existing edges."""
        matrix = torch.zeros((n_rows,n_rows))
        r = self.full_adj[0]
        c = self.full_adj[1]
        matrix[r,c] = vector
        symm_mat = torch.tril(matrix) + torch.tril(matrix,-1)
        return symm_mat
    

class GAALVExplModule(torch.nn.Module):
    """Class for the GCN-conv explanation module of CFPG-v.2"""
    def __init__(self, 
            in_feats: int, 
            enc_hidden: int=20,
            dec_hidden: int=64,
            dropout: float=0.2, 
            edge_repr: bool=True,
            device: str="cpu"
        ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.enc_h    = enc_hidden
        self.dec_h    = dec_hidden
        self.device   = device
        self.dropout  = dropout
        self.enc_out_dim = self.enc_h*3 if edge_repr else self.enc_h

        self.Normal = torch.distributions.Normal(0, 1)

        self.enc_gc1 = GCNConv(self.in_feats, self.enc_h)
        
        self.enc_fc_mu  = torch.nn.Linear(self.enc_out_dim, self.enc_h)
        self.enc_fc_var = torch.nn.Linear(self.enc_out_dim, self.enc_h)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.enc_h, self.dec_h),
            torch.nn.ReLU(),
            #torch.nn.Linear(self.dec_h, self.dec_h),
            #torch.nn.ReLU(),
            torch.nn.Linear(self.dec_h, 1),
            torch.nn.ReLU()
        ).to(self.device)

        ## possible decoders
        """n_heads = 7
        #self.decoder = nn.Sequential(        # ZAVVE
        #    nn.Linear(self.expl_embedding, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, n_heads),
        #    nn.LeakyReLU(),
        #    #nn.Softmax(dim=1),
        #    nn.AvgPool1d(n_heads),
        #).to(self.device)

        #self.decoder = nn.Sequential(
        #    nn.Linear(self.expl_embedding, n_heads),
        #    nn.LeakyReLU(),
        #    nn.Softmax(dim=1),
        #    nn.AvgPool1d(n_heads),
        #).to(self.device)"""

    def forward(self, x, edge_index, node_id, bias: float= 0.0, train: bool=True):
        """Forward step with a GCN encoder."""
        # encoder step
        x1 = F.relu(self.enc_gc1(x, edge_index))
        # get edge representation
        x2 = _get_edge_repr(edge_index, x1, node_id)
        
        # compute KLdiv loss
        mu =  self.enc_fc_mu(x2)
        sigma = torch.exp(self.enc_fc_var(x2))
        z = mu + sigma*self.Normal.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        # decoder step
        out_dec = self.decoder(z)
        sampled_mask = _sample_graph(out_dec, bias=bias, training=train)

        return sampled_mask

