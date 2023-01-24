from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
import torch_geometric as ptgeom

from .BaseExplainer import BaseExplainer
from utils.graphs import index_edge

class CFPGExplainer(BaseExplainer):
    """A class encaptulating CF-PGExplainer (Parametrized-CFExplainer).
    
    Args:
        `model_to_explain` (torch.nn.Module): GNN model who's predictions we 
            wish to explain.
        `graphs` (Tensor): the collections of edge_indices representing the graphs.
        `features` (Tensor): the collection of features for each node in the graphs.
        `task` (string): "node" or "graph".
        `epochs` (int): amount of epochs to train our explainer.
        `lr` (float): learning rate used in the training of the explainer.
    
    Methods:
        `_create_explainer_input`: utility;
        `_sample_graph`: utility; sample an explanatory subgraph;
        `_loss`: calculate the loss of the explainer during training;
        `_train`: train the explainer;
        `prepare`: prepare the explanation method for explaining;
        `explain`: search for the subgraph which contributes most to the clasification
             decision of the model-to-be-explained.
    """
    ## default values for explainer parameters
    coeffs = {
        "reg_size": 0.05,
        "reg_ent" : 1.0,
        "reg_cf"  : 5.8, 
        "temp": [5.0, 2.0],
        "sample_bias": 0.5,
    }

    def __init__(self, 
            model_to_explain: torch.nn.Module, 
            edge_index: torch.Tensor, 
            features: torch.Tensor, 
            task: str="node", 
            epochs: int=30, 
            lr: float=0.003, 
            **kwargs
        ):
        super().__init__(model_to_explain, edge_index, features, task)
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
            node_embed = embeds[node_id].repeat(rows.size(0), 1)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1)
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

    def _loss(self, masked_pred, original_pred, mask):
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
        #if not pred_same: print("CF example found", pred_same)
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
            embeds = self.model_to_explain.embedding(self.features, self.adj).detach()
            

        # Start training loop
        with tqdm(range(0, self.epochs), desc="[CF-PGExplainer]> ...training", disable=False) as epochs_bar:
            for e in epochs_bar:
                optimizer.zero_grad()
                loss_total = torch.FloatTensor([0]).detach()
                size_total = torch.FloatTensor([0]).detach()
                ent_total  = torch.FloatTensor([0]).detach()
                pred_total = torch.FloatTensor([0]).detach()
                t = temp_schedule(e)

                for n in indices:
                    n = int(n)
                    #print(n)
                    if self.type == 'node':
                        # Similar to the original paper we only consider a subgraph for explaining
                        feats = self.features
                        graph = ptgeom.utils.k_hop_subgraph(n, 3, self.adj)[1]
                    else:
                        feats = self.features[n].detach()
                        graph = self.adj[n].detach()
                        embeds = self.model_to_explain.embedding(feats, graph).detach()

                    # Sample possible explanation
                    input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                    #print("embeds :", embeds.size())
                    #print("input_expl :", input_expl.size())

                    sampling_weights = self.explainer_mlp(input_expl)
                    mask = self._sample_graph(sampling_weights, t, bias=sample_bias).squeeze()
                    #print("sampling_weights :", sampling_weights.size())
                    #print("mask             :", mask.size())

                    masked_pred = self.model_to_explain(feats, graph, edge_weights=mask)
                    original_pred = self.model_to_explain(feats, graph)

                    if self.type == 'node': # we only care for the prediction of the node
                        masked_pred = masked_pred[n]#.unsqueeze(dim=0)
                        original_pred = original_pred[n].argmax()
                        #print("masked pred:", masked_pred.size())
                        #print("origin pred:", original_pred.size())

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


    def prepare(self, indices=None):
        """
        Prepars the explanation method for explaining. When using a parametrized 
        explainer like PGExplainer, we first need to train the explainer MLP.

        Args
        - `indices` : Indices over which we wish to train.
        """
        if indices is None: # Consider all indices
            indices = range(0, self.adj.size(0))

        self._train(indices=torch.Tensor(indices))

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
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.adj)[1]
            embeds = self.model_to_explain.embedding(self.features, self.adj).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.adj[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph).detach()

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
