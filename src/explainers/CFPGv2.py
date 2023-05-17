from tqdm import tqdm
from numpy import Inf

import torch
from torch import nn
from torch.optim import Adam, SGD

import torch_geometric
from torch_geometric.utils import k_hop_subgraph 
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, GATv2Conv

from .BaseExplainer import BaseExplainer
from utils.graphs import index_edge


NODE_BATCH_SIZE = 32


class CFPGv2ExplModule(torch.nn.Module):
    """Class for the explanation module of CFPG-v.2"""
    def __init__(self, 
            in_feats: int, 
            enc_hidden: int=20,
            dec_hidden: int=64, 
            conv: str="GCN",
            device: str="cpu"
        ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.enc_h = enc_hidden
        self.dec_h = dec_hidden
        self.device = device

        if conv == "GCN":
            self.enc_gc1 = GCNConv(self.in_feats, self.enc_h)
        elif conv == "GAT":
            self.enc_gc1 = GATv2Conv(self.in_feats, self.enc_h, heads=3, concat=False)
            #self.enc_gc1 = GATv2Conv(self.in_feats, self.enc_h, heads=1)

        self.latent_dim = self.enc_h*3
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.dec_h),
            nn.ReLU(),
            nn.Linear(self.dec_h, 1),
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

    def forward(self, x, edge_index, node_id, bias: float=0.0, train: bool=True):
        # encoder step
        x1 = nn.functional.relu(self.enc_gc1(x, edge_index))

        # parse latent representation
        z = self._latent_edge_repr(edge_index, x1, node_id)

        # decoder step
        x2 = self.decoder(z)
        sampled_mask = self._sample_graph(x2, bias=bias, training=train)

        return sampled_mask

    def _latent_edge_repr(self, sub_index, enc_embeds, node_id):
        """Use encoder node embeddings to create encoder edge embeddings,
        getting each edge embed by concatenating the embeddings of the nodes 
        adjacent to it with the embeddig of `node_id`, the node which we 
        wish to explain.   

        ### Args
        `sub_index`: `torch.Tensor`
            edge_index of the neighborhood subgraph of `node_id`;

        `enc_embeds`: `torch.Tensor`
            embedding of all nodes in `node_id` neighborhood
        
        `node_id`: int
            id of the node we wish to explain
        
        ### Returns
            Concatenated encoder edge embeddings
        """
        rows = sub_index[0]
        cols = sub_index[1]
        row_embeds = enc_embeds[rows]
        col_embeds = enc_embeds[cols]

        node_embed = enc_embeds[node_id].repeat(rows.size(0), 1).to(self.device)
        parsed_rep = torch.cat([row_embeds, col_embeds, node_embed], 1).to(self.device)
        
        return parsed_rep
    
    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
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
            gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(self.device)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph



class CFPGv2(BaseExplainer):
    """A class encaptulating CF-PGExplainer v.2 (Counterfactual-PGExplainer)"""
    coeffs = {            ## default values for explainer parameters
        "lr": 0.003,
        "reg_size": 0.5,
        "reg_ent" : 1.0,
        "reg_cf"  : 5.0, 
        "temps": [5.0, 2.0],
        "sample_bias": 0.0,
    }

    def __init__(self, 
            model_to_explain: torch.nn.Module, 
            data_graph: torch_geometric.data.Data,
            conv: str="GCN",
            task: str="node", 
            epochs: int=30, 
            device: str="cpu",
            coeffs: dict=None
        ):
        """### Args
        `model_to_explain` : torch.nn.Module
            GNN model who's predictions we wish to explain.

        `data_graph` : torch_geometric.data.Data
            the collections of edge_indices representing the graphs.

        `task` : string
            classification task, "node" or "graph".

        `epochs` : int
            amount of epochs to train our explainer.

        `coeffs` : dict
            a dict containing parameters for training the explainer (e.g.
            lr, temprature, etc..).
        """
        super().__init__(model_to_explain, data_graph, task, device)
        self.expl_name = "CFPG-v2"
        self.adj = self.data_graph.edge_index.to(device)
        self.features = self.data_graph.x.to(device)
        self.epochs = epochs
        for k,v in coeffs.items():
            self.coeffs[k] = v
        print("\t>> explainer:", self.expl_name)
        print("\t>> coeffs:", self.coeffs)

        if self.type == "graph": # graph classificatio model
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3

        # Instantiate the explainer model
        in_feats = self.model_to_explain.embedding_size
        self.explainer_module = CFPGv2ExplModule(in_feats,20,64,conv,device)

    def loss(self, masked_pred: torch.Tensor, original_pred: torch.Tensor, mask: torch.Tensor):
        """
        Returns the loss score based on the given mask.

        ### Args:
        `masked_pred` : torch.Tensor
            Prediction based on the current explanation

        `original_pred` : torch.Tensor
            Predicion based on the original graph

        `edge_mask` : torch.Tensor
            Current explanaiton

        `reg_coefs` : torch.Tensor
            regularization coefficients

        ### Return
            Tuple of Tensors (loss,size_loss,mask_ent_loss,pred_loss)
        """
        reg_size = self.coeffs["reg_size"]
        reg_ent  = self.coeffs["reg_ent"]
        reg_cf   = self.coeffs["reg_cf"]
        EPS = 1e-15

        # Regularization losses
        mask_mean = mask.mean()
        #cf_edges = (mask > mask_mean).sum()
        #tot_edges = torch.ones(mask.size()).to(self.device).sum()
        #size_loss = ((tot_edges - cf_edges).abs()) / 2

        #size_loss = -((mask > mask_mean)).sum()   # working fine
        #print("\t>> cf mask size:", size_loss.item())      # working fine
        size_loss = -(mask.sigmoid()).sum()     # old
        size_loss = size_loss * reg_size

        #scale = 0.99
        #mask = mask*(2*scale-1.0)+(1.0-scale)

        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = reg_ent * torch.mean(mask_ent_reg)

        # Explanation loss
        pred_same = (masked_pred.argmax().item() == original_pred).float()
        #cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        cce_loss = torch.nn.functional.nll_loss(masked_pred, original_pred)
        pred_loss = pred_same * (-1 * cce_loss) * reg_cf

        # ZAVVE: TODO tryin' to optimize objective function for cf case
        loss_total = size_loss + pred_loss + mask_ent_loss
        return loss_total, size_loss, mask_ent_loss, pred_loss

    def _train(self, indices=None):
        """
        Main method to train the modeledge_weights=None
        
        Args: 
        - indices: Indices that we want to use for training.
        """
        lr = self.coeffs["lr"]
        temp = self.coeffs["temps"]
        sample_bias = self.coeffs["sample_bias"]

        # Make sure the explainer model can be trained
        self.explainer_module.train()

        # Create optimizer and temperature schedule
        opt = self.coeffs["opt"]
        if opt == "Adam": 
            optimizer = Adam(self.explainer_module.parameters(), lr=lr)
        elif opt == "SGD":
            #optimizer = SGD(self.explainer_mlp.parameters(), lr=lr, nesterov=True, momentum=0.9)
            optimizer = SGD(self.explainer_module.parameters(), lr=lr)

        temp_schedule = lambda e: temp[0]*((temp[1]/temp[0])**(e/self.epochs))

        ## If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.adj)[0].detach().to(self.device)

        # use NeighborLoader to sample batch_size nodes and their respective 3-hop neighborhood
        gnn_iter = 3               # GCN model has 3 mp iteration
        num_neighbors = -1         # -1 for all neighbors
        loader = NeighborLoader(
            self.data_graph,
            num_neighbors=[num_neighbors] * gnn_iter,          
            batch_size=NODE_BATCH_SIZE, 
            input_nodes=indices.cpu(),
            disjoint=False,
        )
        n_indices = indices.size(0)
        n_batches = len(loader)

        self.cf_examples = {}
        best_loss = Inf
        # Start training loop
        with tqdm(range(0, self.epochs), desc=f"[{self.expl_name}]> training", disable=False) as epochs_bar:
            for e in epochs_bar:
                optimizer.zero_grad()
                loss_total = torch.FloatTensor([0]).detach().to(self.device)
                size_total = torch.FloatTensor([0]).detach().to(self.device)
                ent_total  = torch.FloatTensor([0]).detach().to(self.device)
                pred_total = torch.FloatTensor([0]).detach().to(self.device)
                t = temp_schedule(e)
                
                b_id = 0
                for node_batch in loader:
                    if b_id == (n_batches-1):   # last batch may be smaller
                        curr_batch_size = n_indices % NODE_BATCH_SIZE
                    else:
                        curr_batch_size = NODE_BATCH_SIZE

                    #if self.type == 'node':
                    batch_feats  = node_batch.x
                    batch_graph  = node_batch.edge_index
                    global_n_ids = node_batch.n_id
                    b_id += 1

                    for b_n_idx in range(curr_batch_size):
                        # only need node_id neighnbors to compute the explainer input
                        global_idx = global_n_ids[b_n_idx].to(self.device)

                        # we only consider 3hop-neighborhood for explaining
                        sub_nodes, sub_index, n_map, _ = k_hop_subgraph(b_n_idx, 3, batch_graph, relabel_nodes=True)
                        sub_index = sub_index.to(self.device)
                        sub_feats = batch_feats[sub_nodes, :].to(self.device)

                        # relabel sub-graph with global node indices   # still needed?
                        global_n_ids = global_n_ids.to(self.device)
                        sub_graph = torch.take(global_n_ids,sub_index)    
                        
                        # compute explanation mask
                        expl_feats = embeds[global_n_ids].to(self.device)
                        #print("\n\t>> expl feats:", expl_feats.size())
                        #print("\t>> sub feats:", sub_feats.size())
                        #print("\t>> sub graph:", sub_index.size())
                        mask = self.explainer_module(expl_feats, sub_index, n_map, bias=sample_bias)
                        #exit(0)

                        masked_pred, cf_feat = self.model_to_explain(sub_feats, sub_index, edge_weights=mask, cf_expl=True)
                        original_pred = self.model_to_explain(sub_feats, sub_index)

                        sub_node_idx = n_map.item()
                        if self.type == 'node': # node class prediction
                            # when considering the features subset, node prediction is at index 0
                            masked_pred = masked_pred[sub_node_idx]
                            original_pred = original_pred[sub_node_idx].argmax()
                            pred_same = (masked_pred.argmax() == original_pred)

                        id_loss, size_loss, ent_loss, pred_loss = self.loss(masked_pred=masked_pred, 
                                                                    original_pred=original_pred, 
                                                                    mask=mask)  #mask


                        # if masked prediction is different from original, save the CF example
                        if pred_same == 0:
                            #print("cf example found for node", global_idx)
                            best_loss = id_loss
                            cf_ex = {"best_loss": best_loss, "mask": mask, "feats": cf_feat[sub_node_idx]}
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


    def prepare(self, indices=None):
        """Prepars the explanation method for explaining. When using a parametrized 
        explainer like PGExplainer, we first need to train the explainer MLP.

        ### Args
        `indices` : list
            node indices over which we wish to train.
        """
        if indices is None: # Consider all indices
            indices = range(0, self.adj.size(0))

        indices = torch.LongTensor(indices).to(self.device)   
        self._train(indices=indices)

    def explain(self, index):
        """Given the index of a node/graph this method returns its explanation. 
        This only gives sensible results if the prepare method has already been called.

        ### Args
        index : `int`
            index of the node/graph that we wish to explain

        ### Return
            explanation graph and edge weights
        """
        index = int(index)
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            sub_nodes, sub_graph, n_map, _ = k_hop_subgraph(index, 3, self.adj, relabel_nodes=False)
            embeds = self.model_to_explain.embedding(self.features, self.adj)[0].detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.adj[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph)[0].detach()

        # Use explainer mlp to get an explanation
        expl_feats = embeds[sub_nodes, :].to(self.device)
        #mask = self.explainer_module(sub_feats, sub_graph, n_map)
        mask = self.explainer_module(embeds, sub_graph, index, train=False)
        
        # to get opposite of cf-mask, i.e. explanation
        cf_adj = torch.ones(mask.size()).to(self.device) 
        mask = (cf_adj - mask).abs()
        #print("\n\t>> mask:", mask.size())
        #print("\t>> mean:", mask.mean())
        #print("\t>> sum :", (mask > 0.5).sum())
        #exit(0)
        #mask = (mask > 0.5).float()

        expl_graph_weights = torch.zeros(sub_graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = sub_graph.T[i]
            t = index_edge(sub_graph, pair)
            expl_graph_weights[t] = mask[i]

        return sub_graph, expl_graph_weights
