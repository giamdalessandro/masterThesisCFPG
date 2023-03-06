from tqdm import tqdm
from numpy import Inf

import torch
from torch import nn
from torch.optim import Adam

import torch_geometric
from torch_geometric.utils import k_hop_subgraph 
from torch_geometric.loader import NeighborLoader

from .BaseExplainer import BaseExplainer
from utils.graphs import index_edge


NODE_BATCH_SIZE = 64


class CFPGExplainer(BaseExplainer):
    """A class encaptulating CF-PGExplainer (Parametrized-CFExplainer).
    
    Methods:
        `prepare`: prepare the explanation method for explaining;
        `explain`: search for the subgraph which contributes most to the clasification
             decision of the model-to-be-explained.
    """
    ## default values for explainer parameters
    coeffs = {
        "reg_size": 0.05,
        "reg_ent" : 1.0,
        "reg_cf"  : 5.0, 
        "temp": [5.0, 2.0],
        "sample_bias": 0.0,
    }

    def __init__(self, 
            model_to_explain: torch.nn.Module, 
            data_graph: torch_geometric.data.Data,
            task: str="node", 
            epochs: int=30, 
            lr: float=0.005, 
            device: str="cpu",
            **kwargs
        ):
        """
        `model_to_explain` (torch.nn.Module): GNN model who's predictions we 
            wish to explain.
        `graphs` (Tensor): the collections of edge_indices representing the graphs.
        `features` (Tensor): the collection of features for each node in the graphs.
        `task` (string): "node" or "graph".
        `epochs` (int): amount of epochs to train our explainer.
        `lr` (float): learning rate used in the training of the explainer.
        """
        super().__init__(model_to_explain, data_graph, task, device)
        self.expl_name = "CF-PGExplainer"
        self.adj = self.data_graph.edge_index.to(device)
        self.features = self.data_graph.x.to(device)
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        if self.type == "graph": # graph classificatio model
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3

        # Instantiate the explainer model
        self.explainer_mlp = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _create_explainer_input(self, pair, embeds, node_id):
        """
        Given the embeddign of the sample by the model that we wish to explain, 
        this method construct the input to the mlp explainer model. Depending on
        if the task is to explain a graph or a sample, this is done by either 
        concatenating two or three embeddings.
        
        Args:
            `pair`: edge pair;
            `embeds`: embedding of all nodes in the graph
            `node_id`: id of the node, not used for graph datasets
        
        Returns
            Concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        if self.type == 'node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1).to(self.device)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1).to(self.device)
            #print(">> node id:", node_id)
            #print("\tnode embed:", node_embed.size())
            #print("\trow embed :", row_embeds.size())
            #print("\tcol embed :", col_embeds.size())
            #print("\tpair:", pair.size())
            #exit(0)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1).to(self.device)
        return input_expl

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        r"""
        Implementation of the reparamerization trick to obtain a sample 
        graph while maintaining the posibility to backprop.
        
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
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def loss(self, masked_pred: torch.Tensor, original_pred: torch.Tensor, mask: torch.Tensor):
        """
        Returns the loss score based on the given mask.

        Args:
        - `masked_pred`   : Prediction based on the current explanation
        - `original_pred` : Predicion based on the original graph
        - `edge_mask`     : Current explanaiton
        - `reg_coefs`     : regularization coefficients

        Return
            Tuple of Tensors (loss,size_loss,mask_ent_loss,pred_loss)
        """
        reg_size = self.coeffs["reg_size"]
        reg_ent  = self.coeffs["reg_ent"]
        reg_cf   = self.coeffs["reg_cf"]
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(mask)
        size_loss = torch.sum(mask) * reg_size
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = reg_ent * torch.mean(mask_ent_reg)

        # Explanation loss
        pred_same = (masked_pred.argmax() == original_pred).float()
        #if not pred_same: print("pred_same_:", pred_same)
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        pred_loss = pred_same * (-cce_loss) * reg_cf

        # ZAVVE: TODO tryin' to optimize objective function for cf case
        loss_total = size_loss + mask_ent_loss + pred_loss
        return loss_total, size_loss, mask_ent_loss, pred_loss

    def _train(self, indices=None):
        """
        Main method to train the model
        
        Args: 
        - indices: Indices that we want to use for training.
        """
        temp = self.coeffs["temp"]
        sample_bias = self.coeffs["sample_bias"]

        # Make sure the explainer model can be trained
        self.explainer_mlp.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_mlp.parameters(), lr=self.lr)
        temp_schedule = lambda e: temp[0]*((temp[1]/temp[0])**(e/self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.adj)[0].detach().to(self.device)

        # use NeighborLoader to consider batch_size nodes and their respective neighborhood
        loader = NeighborLoader(
            self.data_graph,
            # Sample n neighbors for each node for 3 GNN iterations, 
            num_neighbors=[-1] * 3,          # -1 for all neighbors
            batch_size=NODE_BATCH_SIZE,      # num of nodes in the batch
            input_nodes=indices,
            disjoint=True,
        )

        self.cf_examples = {}
        best_loss = Inf
        # Start training loop
        with tqdm(range(0, self.epochs), desc="[CF-PGExplainer]> ...training", disable=False) as epochs_bar:
            for e in epochs_bar:
                optimizer.zero_grad()
                loss_total = torch.FloatTensor([0]).detach().to(self.device)
                size_total = torch.FloatTensor([0]).detach().to(self.device)
                ent_total  = torch.FloatTensor([0]).detach().to(self.device)
                pred_total = torch.FloatTensor([0]).detach().to(self.device)
                t = temp_schedule(e)

                #for idx in indices:
                #    idx = int(idx)
                #    #print(idx)
                #    if self.type == 'node':
                #        # Similar to the original paper we only consider a subgraph for explaining
                #        feats = self.features
                #        sub_graph = k_hop_subgraph(idx, 3, self.adj)[1]
                #    else:
                #        feats = self.features[idx].detach()
                #        graph = self.adj[idx].detach()
                #        embeds = self.model_to_explain.embedding(feats, graph)[0].detach()
                
                for node_batch in loader:
                    if self.type == 'node':
                        # Similar to the original paper we only consider a subgraph for explaining
                        feats = node_batch.x
                        graph = node_batch.edge_index
                        batch_ids = node_batch.batch 
                        #print("\n\tbatch feats:", feats.size())
                        #print("\tbatch graph:", graph.size())
                        #print("\tbatch n_id:", node_batch.n_id)

                        # NeighborLoader may include random nodes to match the chosen batch_size,
                        # may need to consider only a subset of the batch  
                        curr_batch_size = NODE_BATCH_SIZE
                        if NODE_BATCH_SIZE > 1:
                            valid_nodes = torch.argwhere(torch.where(batch_ids[:NODE_BATCH_SIZE] == 0, 1, 0)).squeeze()
                            if valid_nodes.nelement() > 1:
                                #print(">> curr batch size:", valid_nodes[1])
                                curr_batch_size = valid_nodes[1].item()
 
                    else: 
                        raise NotImplementedError("graph classification")   # graph classification case


                    for b_idx in range(curr_batch_size):
                        # only need node_id neighnbors to compute the explainer input
                        global_idx = node_batch.n_id[b_idx].item()
                        #print("\n------- node (global_id)", global_idx, f"(local id {b_idx})")

                        neighbors = torch.argwhere(torch.where(batch_ids == b_idx, 1, 0)).squeeze()
                        if neighbors.nelement() > 1: 
                            sub_feats = torch.stack([feats[n] for n in neighbors])
                        else:
                            print("\n\tneighbors:", neighbors)
                            print("\tlocal id:", b_idx, "global id:", global_idx)
                            exit(0)

                        sub_graph = k_hop_subgraph(b_idx, 3, graph, relabel_nodes=True)[1]
                        #print("\t>> sub_feats:", sub_feats.size())
                        #print("\t>> sub_graph:", sub_graph.size())
                        
                        # possible explanation for each node in sample
                        input_expl = self._create_explainer_input(sub_graph, embeds, global_idx).unsqueeze(0)

                        sampling_weights = self.explainer_mlp(input_expl)
                        mask = self._sample_graph(sampling_weights, t, bias=sample_bias).squeeze()

                        masked_pred, cf_feat = self.model_to_explain(sub_feats, sub_graph, edge_weights=mask, cf_expl=True)
                        original_pred = self.model_to_explain(sub_feats, sub_graph)

                        sub_node_idx = 0
                        if self.type == 'node': # node class prediction
                            # when considering the features subset, node prediction is at index 0
                            masked_pred = masked_pred[sub_node_idx]
                            original_pred = original_pred[sub_node_idx].argmax()
                            pred_same = (masked_pred.argmax() == original_pred)

                        id_loss, size_loss, ent_loss, pred_loss = self.loss(masked_pred=masked_pred, 
                                                                    original_pred=original_pred, 
                                                                    mask=mask)

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

                    epochs_bar.set_postfix(loss=f"{loss_total.item():.4f}", size_loss=f"{size_total.item():.4f}",
                                        ent_loss=f"{ent_total.item():.4f}", pred_loss=f"{pred_total.item():.4f}")

                loss_total.backward()
                optimizer.step()


    def prepare(self, indices=None):
        """
        Prepars the explanation method for explaining. When using a parametrized 
        explainer like PGExplainer, we first need to train the explainer MLP.

        Args
        - `indices` : Indices over which we wish to train.
        """
        if indices is None: # Consider all indices
            indices = range(0, self.adj.size(0))

        indices = torch.LongTensor(indices).to(self.device)   
        self._train(indices=indices)

    def explain(self, index):
        """
        Given the index of a node/graph this method returns its explanation. 
        This only gives sensible results if the prepare method has already been called.

        Args
        - index: index of the node/graph that we wish to explain

        Return
            explanaiton graph and edge weights
        """
        index = int(index)
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph = k_hop_subgraph(index, 3, self.adj)[1]
            embeds = self.model_to_explain.embedding(self.features, self.adj)[0].detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.adj[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph)[0].detach()

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
