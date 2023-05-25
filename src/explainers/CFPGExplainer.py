from tqdm import tqdm
from numpy import Inf

import torch
from torch import nn
from torch.optim import Adam, SGD

import torch_geometric
from torch_geometric.utils import k_hop_subgraph 
from torch_geometric.loader import NeighborLoader

from .BaseExplainer import BaseExplainer
from utils.graphs import index_edge


NODE_BATCH_SIZE = 32


class CFPGExplainer(BaseExplainer):
    """A class encaptulating CF-PGExplainer (Counterfactual-PGExplainer).
    
    Methods:
        `prepare`: prepare the explanation method for explaining;
        `explain`: search for the subgraph which contributes most to the clasification
             decision of the model-to-be-explained.
    """
    ## default values for explainer parameters
    coeffs = {
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
        self.expl_name = "CFPG"
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
        self.explainer_mlp = nn.Sequential(         # PGE default
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)


    def _create_explainer_input(self, pair, embeds, node_id):
        """Given the embeddign of the sample by the model that we wish to explain, 
        this method construct the input to the mlp explainer model. Depending on
        if the task is to explain a graph or a sample, this is done by either 
        concatenating two or three embeddings.
        
        ### Args
        `pair`: edge pair;

        `embeds`: embedding of all nodes in the graph
        
        `node_id`: id of the node, not used for graph datasets
        
        ### Returns
            Concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]

        if self.type == 'node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1).to(self.device)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1).to(self.device)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1).to(self.device)
        return input_expl

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

    def loss(self, masked_pred: torch.Tensor, original_pred: torch.Tensor, mask: torch.Tensor):
        """Returns the loss score based on the given mask.

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

        size_loss = -((mask > mask_mean)).sum()   # working fine
        #size_loss = (mask.sigmoid()).sum()     # old
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
        """Main method to train the model
        
        Args: 
        - indices: Indices that we want to use for training.
        """
        lr = self.coeffs["lr"]
        temp = self.coeffs["temps"]
        sample_bias = self.coeffs["sample_bias"]

        # Make sure the explainer model can be trained
        self.explainer_mlp.train()

        # Create optimizer and temperature schedule
        opt = self.coeffs["opt"]
        if opt == "Adam": 
            optimizer = Adam(self.explainer_mlp.parameters(), lr=lr)
        elif opt == "SGD":
            #optimizer = SGD(self.explainer_mlp.parameters(), lr=lr, nesterov=True, momentum=0.9)
            optimizer = SGD(self.explainer_mlp.parameters(), lr=lr)

        temp_schedule = lambda e: temp[0]*((temp[1]/temp[0])**(e/self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
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

        self.history = {}
        epoch_loss_tot  = []
        epoch_loss_size = []
        epoch_loss_ent  = []
        epoch_loss_pred = []
        epoch_cf_ex     = []

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

                    if self.type == 'node':
                        # Similar to the original paper we only consider a subgraph for explaining
                        batch_feats  = node_batch.x
                        batch_graph  = node_batch.edge_index
                        global_n_ids = node_batch.n_id
                        b_id += 1

                    for b_n_idx in range(curr_batch_size):
                        # only need node_id neighnbors to compute the explainer input
                        global_idx = global_n_ids[b_n_idx].to(self.device)
                        #print("\n------- node (global_id)", global_idx, f"(local id {b_n_idx})")

                        sub_nodes, sub_index, n_map, _ = k_hop_subgraph(b_n_idx, 3, batch_graph, relabel_nodes=True)
                        sub_index = sub_index.to(self.device)
                        sub_feats = batch_feats[sub_nodes, :].to(self.device)
                        #print("\t>> sub_feats:", sub_feats.size())

                        global_n_ids = global_n_ids.to(self.device)
                        sub_graph = torch.take(global_n_ids,sub_index)    # global node indices to sub-graph
                        #print("\t>> sub_index:", sub_index.size())
                        
                        # compute edge embeddings to be fed to the explainer
                        input_expl = self._create_explainer_input(sub_graph, embeds, global_idx).unsqueeze(0)
                        
                        sampling_weights = self.explainer_mlp(input_expl)
                        mask = self._sample_graph(sampling_weights, t, bias=sample_bias, training=False).squeeze()
                        
                        # to get opposite of cf-mask, i.e. explanation
                        cf_adj = torch.ones(mask.size()).to(self.device) 
                        cf_adj = (cf_adj - mask).abs()
                        #print("\t>> cf_adj:", cf_adj.size())
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


                        # if original prediction changes save the CF example
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
                    
                # metrics to plot
                epoch_loss_tot.append(loss_total.item())
                epoch_loss_size.append(size_total.item())
                epoch_loss_ent.append(ent_total.item())
                epoch_loss_pred.append(pred_total.item())
                epoch_cf_ex.append(len(self.cf_examples.keys()))

                loss_total.backward()
                optimizer.step()

        self.history["train_loss"] = {
        "loss_tot"  : epoch_loss_tot,
        "loss_size" : epoch_loss_size,
        "loss_ent"  : epoch_loss_ent,
        "loss_pred" : epoch_loss_pred,
        }
        self.history["cf_fnd"] = epoch_cf_ex
        self.history["cf_tot"] = n_indices


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
        index : int
            index of the node/graph that we wish to explain

        ### Return
            explanation graph and edge weights
        """
        index = int(index)
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            sub_nodes, graph, _, _ = k_hop_subgraph(index, 3, self.adj)
            embeds = self.model_to_explain.embedding(self.features, self.adj)[0].detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.adj[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph)[0].detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, embeds, index).unsqueeze(dim=0)
        sampling_weights = self.explainer_mlp(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
    
        # to get opposite of cf-mask, i.e. explanation
        cf_adj = torch.ones(mask.size()).to(self.device) 
        mask = (cf_adj - mask).abs()

        expl_graph_weights = torch.zeros(graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights #, sub_nodes, mask
