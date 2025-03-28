import imp
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import torch_geometric
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import NeighborLoader

from .BaseExplainer import BaseExplainer
from gnns.CFGNNpaper import GCNSyntheticPerturb

from utils.graphs import index_edge
from utils.graphs import normalize_adj


NODE_BATCH_SIZE = 32


class PCFExplainer(BaseExplainer):
    """Parametrized-CF-GNNExplainer class, computes counterfactual subgraph. 
    Based on CF-GNNexplainer (https://arxiv.org/abs/2102.03322).
	"""
    ## default values for explainer parameters
    coeffs = {
        "reg_size": 0.5,
        "reg_ent" : 1.0,
        "reg_cf"  : 5.0, 
        "temps": [5.0, 2.0],
        "sample_bias": 0.0,
        "n_hid"   : 20,
        "dropout" : 0.0,
        "beta"    : 0.5
    }

    def __init__(self, 
            model: torch.nn.Module, 
            data_graph: torch_geometric.data.Data,
            norm_adj: torch.Tensor, 
            task: str="node",  
            epochs=30, 
            lr=0.003, 
            device: str="cpu",
            coeffs: dict= None
        ):
        super().__init__(model, data_graph, task, device)
        self.expl_name = "PCF"
        self.adj = self.data_graph.edge_index.to(device)
        self.features = self.data_graph.x.to(device)
        self.norm_adj = norm_adj.to(device)
        self.model_to_explain.eval()

        # from config
        self.epochs = epochs
        self.lr     = lr
        for k,v in coeffs.items():
            self.coeffs[k] = v
        print("\t>> explainer:", self.expl_name)
        print("\t>> coeffs:", self.coeffs)

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
        ).to(self.device)
        
    def _cf_prepare(self, n_nodes: int, n_feats: int, sub_index: torch.Tensor, verbose: bool=False):
        """Instantiate GCN Perturbation Model for the explanation. Creates a model
        with the same parameters of the original model to explain, that takes into
        account the perturbation matrix when performing a prediction to compute
        the counterfactual examples. 
        """
        n_hid   = self.coeffs["n_hid"]
        dropout = self.coeffs["dropout"]
        beta    = self.coeffs["beta"]
        nclass  = self.model_to_explain.nclass

        # need dense adjacency matrix for GCNSynthetic model
        v = torch.ones(sub_index.size(1))
        s = (n_nodes,n_nodes)
        dense_index = torch.sparse_coo_tensor(indices=sub_index, values=v, size=s).to_dense()
        norm_sub_adj = normalize_adj(dense_index).to(self.device)

		# Instantiate CF model class, load weights from original model
        cf_model = GCNSyntheticPerturb(    
                            nfeat = n_feats, 
                            nhid=n_hid, 
                            nout=n_hid,
                            nclass=nclass, 
                            adj=norm_sub_adj, 
                            dropout=dropout, 
                            beta=beta,
                            edge_additions=True,
                            device=self.device).to(self.device)
        
        cf_model.load_state_dict(self.model_to_explain.state_dict(), strict=False)

		# Freeze weights from original model in cf_model
        for name, param in cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        if verbose:
            for name, param in self.model_to_explain.named_parameters():
                print("orig model requires_grad: ", name, param.requires_grad)
            for name, param in cf_model.named_parameters():
                print("cf model requires_grad: ", name, param.requires_grad)

        return cf_model
        
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

        if self.type == 'node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl

    def _sample_graph(self, sampling_weights, temperature: float=1.0, bias: float=0.0, training: bool=True):
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
            gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(self.device)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def loss(self, masked_pred: torch.Tensor, original_pred: torch.Tensor, mask: torch.Tensor):
        """ TODO: Returns the explainer MLP loss score based on the given mask.

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
        #mask = torch.sigmoid(mask)
        mask_mean = torch.mean(mask)
        size_loss = -((mask > mask_mean)).sum()
        #print("mask size:", size_loss)
        #size_loss = (torch.sum(self.adj) - torch.sum(mask)) * reg_size
        size_loss = size_loss * reg_size
        
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = reg_ent * torch.mean(mask_ent_reg)

        # Countefactual loss
        pred_same = (torch.argmax(masked_pred, dim=1) == original_pred).float()
        #print("pred_same:", pred_same)
        cce_loss = torch.nn.functional.nll_loss(masked_pred, original_pred)
        pred_loss = pred_same * (-cce_loss) * reg_cf
        #print("cce_loss:", cce_loss, "\tpred_loss:", pred_loss)
        
        # ZAVVE: TODO tryin' to optimize objective function for cf case
        loss_total = size_loss + pred_loss + mask_ent_loss
        return loss_total, size_loss, mask_ent_loss, pred_loss

    def _train(self, indices, original_preds: torch.Tensor, verbose: bool=False):
        """Train the explainer MLP.

        Args: 
        - indices: Indices that we want to use for training.
        """
        # Make sure the explainer model can be trained
        lr = self.coeffs["lr"]
        temp = self.coeffs["temps"]
        sample_bias = self.coeffs["sample_bias"]
        self.explainer_mlp.train()
        #print("adj :", self.adj.size())

        # Create optimizer and temperature schedule
        optimizer = optim.Adam(self.explainer_mlp.parameters(), lr=lr)
        #optimizer = optim.SGD(self.explainer_mlp.parameters(), lr=self.lr, momentum=0.2)
        temp_schedule = lambda e: temp[0]*((temp[1]/temp[0])**(e/self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.norm_adj).detach().to(self.device)

        # use NeighborLoader to consider batch_size nodes and their respective neighborhood
        loader = NeighborLoader(
            self.data_graph,
            # Sample n neighbors for each node for 3 GNN iterations, 
            num_neighbors=[-1] * 3,          # -1 for all neighbors
            batch_size=NODE_BATCH_SIZE,      # num of nodes in the batch
            input_nodes=indices.cpu(),
            disjoint=False,
        )
        n_indices = indices.size(0)
        n_batches = len(loader)

        self.cf_examples = {}
        best_loss = np.inf
        # explainer training loop
        with tqdm(range(0, self.epochs), desc=f"[{self.expl_name}]> training", disable=False) as epochs_bar:
            for e in epochs_bar:
                optimizer.zero_grad()
                loss_total = torch.FloatTensor([0]).detach().to(self.device)
                size_total = torch.FloatTensor([0]).detach().to(self.device)
                ent_total  = torch.FloatTensor([0]).detach().to(self.device)
                pred_total = torch.FloatTensor([0]).detach().to(self.device)
                t = temp_schedule(e)

                #for idx in indices:
                #    idx = int(idx)
                #    if self.type == 'node':
                #        # Similar to the original paper we only consider a subgraph for explaining
                #        feats = self.features
                #        graph = k_hop_subgraph(idx, 3, self.adj)[1]
                #    else:
                #        feats = self.features[idx].detach()
                #        graph = self.adj[idx].detach()
                #        graph = graph.to_dense()
                #        embeds = self.model_to_explain.embedding(feats, graph).detach()

                b_id = 0
                for node_batch in loader:
                    if b_id == (n_batches-1):     # last batch may be smaller
                        curr_batch_size = n_indices % NODE_BATCH_SIZE
                    else:
                        curr_batch_size = NODE_BATCH_SIZE

                    if self.type == 'node':
                        global_n_ids = node_batch.n_id
                        batch_feats = node_batch.x.to(self.device)
                        batch_graph = node_batch.edge_index
                        b_id += 1

                    # instantiate synthetic perturbation model
                    n_nodes, n_feats = batch_feats.size() 
                    cf_model = self._cf_prepare(n_nodes, n_feats, batch_graph)

                    for b_idx in range(curr_batch_size):
                        # only need node_id neighnbors to compute the explainer input
                        global_idx = global_n_ids[b_idx].to(self.device)
                        #print("\n------- node (global_id)", global_idx, f"(local id {b_idx})")

                        _, sub_index, _ , _ = k_hop_subgraph(b_idx, 3, batch_graph, relabel_nodes=True)
                        #sub_feats = batch_feats[sub_nodes, :]
                        sub_index = sub_index.to(self.device)
                        global_n_ids = global_n_ids.to(self.device)
                        sub_graph = torch.take(global_n_ids,sub_index).to(self.device)   # global node indices to sub-graph
                        # instantiate synthetic perturbation model
                        #n_nodes, n_feats = sub_feats.size()
                        #cf_model = self._cf_prepare(n_nodes, n_feats, sub_index)

                        # Sample possible explanation
                        input_expl = self._create_explainer_input(sub_graph, embeds, global_idx).unsqueeze(0)

                        sampling_weights = self.explainer_mlp(input_expl)
                        mask = self._sample_graph(sampling_weights, t, bias=sample_bias).squeeze()
                        
                        s = (n_nodes,n_nodes)
                        #cf_adj = torch.ones(mask.size()).to(self.device) 
                        #cf_adj = (cf_adj - mask).abs()
                        dense_mask = torch.sparse_coo_tensor(indices=sub_index, values=mask, size=s).to_dense()
                        
                        #original_pred = self.model_to_explain.forward(self.features, adj=self.norm_adj)
                        #batch_feats = batch_feats.to(self.device)
                        dense_mask = dense_mask.to(self.device)
                        masked_pred, cf_P, cf_feats = cf_model.forward_prediction(batch_feats, P_mask=dense_mask)
                        #masked_pred = masked_pred.cpu()

                        sub_node_idx = b_idx
                        if self.type == 'node':    # we only care for the prediction of the node
                            masked_pred = masked_pred[sub_node_idx].unsqueeze(dim=0)
                            original_pred = torch.argmax(original_preds[global_idx.item()]).unsqueeze(0)       
                            pred_same = (torch.argmax(masked_pred, dim=1) == original_pred)
                            
                        id_loss, size_loss, ent_loss, pred_loss = self.loss(masked_pred=masked_pred, 
                                                                original_pred=original_pred, 
                                                                mask=mask) #mask

                        # if original prediction changes save the CF example
                        pred_same = (torch.argmax(masked_pred, dim=1) == original_pred)
                        if pred_same == 0: 
                            #print("cf example found for node", idx)
                            best_loss = id_loss
                            cf_ex = {"best_loss": best_loss,"mask": cf_P, "feats": cf_feats[sub_node_idx]}
                            try: 
                                if best_loss < self.cf_examples[str(global_idx)]["best_loss"]:
                                    self.cf_examples[str(global_idx)] = cf_ex
                            except KeyError:
                                self.cf_examples[str(global_idx)] = cf_ex

                        loss_total += id_loss
                        size_total += size_loss
                        ent_total  += ent_loss
                        pred_total += pred_loss


                    epochs_bar.set_postfix(loss=f"{loss_total.item():.4f}", l_size=f"{size_total.item():.4f}",
                                         l_ent=f"{ent_total.item():.4f}", l_pred=f"{pred_total.item():.4f}")
                
                loss_total.backward()
                optimizer.step()
              

    def prepare(self, indices):
        """Prepare the PCFExplainer, this happens at every index."""
        if indices is None: # Consider all indices
            indices = range(0, self.norm_adj.size(0))
        indices = torch.LongTensor(indices).to(self.device)

        #print("feats", self.features.device)   
        #print("n_adj", self.norm_adj.device)   
        original_preds = self.model_to_explain.forward(self.features, adj=self.norm_adj).to(self.device)
        self._train(indices=indices, original_preds=original_preds)
        
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
            embeds = self.model_to_explain.embedding(self.features, self.norm_adj).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.adj[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph).detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, embeds, index).unsqueeze(dim=0)
        sampling_weights = self.explainer_mlp(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()

        # to get opposite mask
        cf_adj = torch.ones(mask.size()).to(self.device) 
        mask = (cf_adj - mask).abs()

        expl_graph_weights = torch.zeros(graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights