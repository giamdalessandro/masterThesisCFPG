import imp
import time
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.utils import k_hop_subgraph
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from .BaseExplainer import BaseExplainer
from gnns.CFGNNpaper.gcn_perturb import GCNSyntheticPerturb
from utils.graphs import index_edge, sparse_to_dense_adj



class PCFExplainer(BaseExplainer):
    """Parametrized-CF-GNNExplainer class, computes counterfactual subgraph. 
    Based on CF-GNNexplainer (https://arxiv.org/abs/2102.03322).
	"""
    ## default values for explainer parameters
    coeffs = {
        "reg_size": 0.05,
        "reg_ent" : 1.0,
        "reg_cf"  : 0.75, 
        "temp": [5.0, 2.0],
        "sample_bias": 0.0,
        "n_hid"   : 20,
        "dropout" : 0.0,
        "beta"    : 0.5
    }

    def __init__(self, 
            model: torch.nn.Module, 
            edge_index: torch.Tensor,
            norm_adj: torch.Tensor, 
            features: torch.Tensor, 
            task: str,  
            epochs=30, 
            lr=0.003, 
            device: str="cpu",
            **kwargs
        ):
        super().__init__(model, edge_index, features, task)
        self.device   = device
        self.norm_adj = norm_adj
        self.model    = self.model_to_explain
        self.model.eval()

        # from config
        self.epochs      = epochs
        self.lr          = lr
        self.coeffs.update(kwargs)

        gcn_layers = 3
        n_hid = self.coeffs["n_hid"]
        if self.type == "graph":
            self.expl_embedding = (n_hid*gcn_layers)*2
        else:
            self.expl_embedding = (n_hid*gcn_layers)*3

        # instantiate explainer_mlp;
        self.explainer_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.expl_embedding, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        
    def _cf_prepare(self, verbose: bool=False):
        """Instantiate GCN Perturbation Model for the explanation. Creates a model
        with the same parameters of the original model to explain, that takes into
        account the perturbation matrix when performing a prediction to compute
        the counterfactual examples. 
        """
        n_hid   = self.coeffs["n_hid"]
        dropout = self.coeffs["dropout"]
        beta    = self.coeffs["beta"]
        nclass  = self.model_to_explain.nclass

		# Instantiate CF model class, load weights from original model
        self.cf_model = GCNSyntheticPerturb(    
                            nfeat = self.features.shape[1], 
                            nhid=n_hid, 
                            nout=n_hid,
                            nclass=nclass, 
                            adj=self.norm_adj, 
                            dropout=dropout, 
                            beta=beta,
                            edge_additions=True)
        
        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

		# Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        if verbose:
            for name, param in self.model.named_parameters():
                print("orig model requires_grad: ", name, param.requires_grad)
            for name, param in self.cf_model.named_parameters():
                print("cf model requires_grad: ", name, param.requires_grad)

    def _create_explainer_input(self, pair, embeds, node_id):
        """Given the embedding of the sample by the model that we wish to explain, 
        this method construct the input to the mlp explainer model. Depending on
        if the task is to explain a graph or a sample, this is done by either 
        concatenating two or three embeddings.
        
        Args
        - `pair`    : edge pair
        - `embeds`  : embedding of all nodes in the graph
        - `node_id` : id of the node, not used for graph datasets

        Return
            concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]

        #print("rows :", rows.size())
        #print("cols :", cols.size())
        #print("row_embeds :", row_embeds.size())
        #print("col_embeds :", col_embeds.size())        

        if self.type == 'node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """Implementation of the reparamerization trick to obtain a sample 
        graph while maintaining the possibility to backprop.
        
        Args
        - `sampling_weights` : Weights provided by the mlp;
        - `temperature`      : annealing temperature to make the procedure more deterministic;
        - `bias`             : Bias on the weights to make samplign less deterministic;
        - `training`         : If set to false, the samplign will be entirely deterministic;
        
        Return 
            sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def _loss(self, masked_pred, original_pred, mask):
        """ TODO 
        Returns the loss score based on the given mask.

        Args
        -  `masked_pred`   : Prediction based on the current explanation
        -  `original_pred` : Predicion based on the original graph
        -  `edge_mask`     : Current explanaiton
        -  `reg_coefs`     : regularization coefficients

        Return
            loss
        """
        reg_size = self.coeffs["reg_size"]
        reg_ent  = self.coeffs["reg_ent"]
        reg_cf   = self.coeffs["reg_cf"]
        dist_reg = self.coeffs["beta"]
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(mask)
        #size_loss = (torch.sum(self.adj) - torch.sum(mask)) * reg_size
        size_loss = torch.sum(mask) * reg_size
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = reg_ent * torch.mean(mask_ent_reg)

        # Countefactual loss
        pred_same = (torch.argmax(masked_pred, dim=1) == original_pred).float()
        cce_loss = torch.nn.functional.nll_loss(masked_pred, original_pred)
        pred_loss = pred_same * (-cce_loss) * reg_cf
        #print("cce_loss:", cce_loss, "\tpred_loss:", pred_loss)
        
        # ZAVVE: TODO tryin' to optimize objective function for cf case
        loss_total = size_loss + mask_ent_loss + pred_loss 
        return loss_total, size_loss, mask_ent_loss, pred_loss

    def _train(self, indices, verbose: bool=False):
        """Train the explainer MLP.

        Args: 
        - indices: Indices that we want to use for training.
        """
        # Make sure the explainer model can be trained
        temp = self.coeffs["temp"]
        sample_bias = self.coeffs["sample_bias"]
        self.explainer_mlp.train()
        #print("adj :", self.adj.size())

        # Create optimizer and temperature schedule
        #optimizer = optim.Adam(self.explainer_mlp.parameters(), lr=self.lr)
        optimizer = optim.SGD(self.explainer_mlp.parameters(), lr=self.lr, momentum=0.9)
        temp_schedule = lambda e: temp[0]*((temp[1]/temp[0])**(e/self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model.embedding(self.features, self.norm_adj).detach()

        # explainer training loop
        with tqdm(range(0, self.epochs), desc="[PCFxplainer]> ...training", disable=False) as epochs_bar:
            for e in epochs_bar:
                optimizer.zero_grad()
                loss_total = torch.FloatTensor([0]).detach()
                size_total = torch.FloatTensor([0]).detach()
                ent_total  = torch.FloatTensor([0]).detach()
                pred_total = torch.FloatTensor([0]).detach()
                t = temp_schedule(e)

                for n in indices:
                    n = int(n)
                    if self.type == 'node':
                        # Similar to the original paper we only consider a subgraph for explaining
                        feats = self.features
                        graph = k_hop_subgraph(n, 3, self.adj)[1]
                    else:
                        feats = self.features[n].detach()
                        graph = self.adj[n].detach()
                        graph = graph.to_dense()
                        embeds = self.model.embedding(feats, graph).detach()

                    # Sample possible explanation
                    input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                    #print("sub_graph  :", graph.size())
                    #print("input_expl :", input_expl.size())
                    #print("embeds     :", embeds.size())

                    sampling_weights = self.explainer_mlp(input_expl)
                    mask = self._sample_graph(sampling_weights, t, bias=sample_bias).squeeze()
                    #print("sampling_weights :", sampling_weights.size())
                    #print("mask    :", mask.size())
                    #print("graph   :", graph.size())
                    #print("#edge in expl    :", torch.sum(mask>0.5).item())

                    s = self.norm_adj.size()
                    dense_mask = torch.sparse_coo_tensor(indices=graph, values=mask, size=s).to_dense()
                    #print("mask dense  :", dense_mask.size())
                    #print("adj dense   :", self.adj.size())

                    #masked_pred = self.model.forward(feats, adj=self.adj)
                    original_pred = self.model.forward(feats, adj=self.norm_adj)
                    #masked_pred = self.cf_model.forward(feats, sub_adj=self.norm_adj)
                    masked_pred, _ = self.cf_model.forward_prediction(feats, P_mask=dense_mask)
                    
                    #print("masked_pred  :", torch.argmax(masked_pred[n]).unsqueeze(dim=0), "idx:", n)
                    #print("origin_pred  :", torch.argmax(original_pred[n].unsqueeze(dim=0)))

                    if self.type == 'node': # we only care for the prediction of the node
                        masked_pred = masked_pred[n].unsqueeze(dim=0)
                        original_pred = torch.argmax(original_pred[n]).unsqueeze(0)
                        #original_pred = original_pred[n]

                    id_loss, size_loss, ent_loss, pred_loss = self._loss(masked_pred=masked_pred, 
                                                            original_pred=original_pred, 
                                                            mask=mask)
                    loss_total += id_loss
                    size_total += size_loss
                    ent_total  += ent_loss
                    pred_total += pred_loss

                epochs_bar.set_postfix(loss=f"{loss_total.item():.4f}", size_loss=f"{size_total.item():.4f}",
                                        ent_loss=f"{ent_total.item():.4f}", pred_loss=f"{pred_total.item():.4f}")

                loss_total.backward()
                optimizer.step()
              

    def prepare(self, indices):
        """Prepare the PCFExplainer, this happens at every index."""
        # instantiate synthetic perturbation model
        self._cf_prepare()

        if indices is None: # Consider all indices
            indices = range(0, self.norm_adj.size(0))
        self._train(indices=indices)
        
    def explain(self, index):
        """Given the index of a node/graph this method returns its 
        explanation. This only gives sensible results if the prepare
        method has already been called.

        Args
        - index: index of the node/graph that we wish to explain

        Return
            explanaiton graph and edge weights
        """
        index = int(index)
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph = k_hop_subgraph(index, 3, self.adj)[1]
            embeds = self.model.embedding(self.features, self.norm_adj).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.adj[index].clone().detach()
            embeds = self.model.embedding(feats, graph).detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, embeds, index).unsqueeze(dim=0)
        sampling_weights = self.explainer_mlp(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()

        expl_graph_weights = torch.zeros(graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights