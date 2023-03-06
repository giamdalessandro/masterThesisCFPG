import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm

from .BaseExplainer import BaseExplainer
from utils.graphs import index_edge


class PGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """
    coeffs = {
        "reg_size": 0.05,
        "reg_ent" : 1.0,
        "temp": [5.0, 2.0],
        "sample_bias": 0.0,
    }

    def __init__(self, 
            model_to_explain: torch.nn.Module, 
            data_graph: ptgeom.data.Data,
            #edge_index: torch.Tensor, 
            #features: torch.Tensor, 
            task: str="node", 
            epochs: int=30, 
            lr: float=0.003, 
            **kwargs
        ):
        super().__init__(model_to_explain, data_graph, task)
        self.expl_name = "PGExplainer"
        self.adj = self.data_graph.edge_index
        self.features = self.data_graph.x
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        if self.type == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3

        # Instantiate the explainer model
        self.explainer_model = nn.Sequential(
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
        
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
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
        """
        Implementation of the reparamerization trick to obtain a sample 
        graph while maintaining the posibility to backprop.
        
        Args
        - `sampling_weights` : Weights provided by the mlp
        - `temperature`      : annealing temperature to make the procedure more deterministic
        - `bias`             : Bias on the weights to make samplign less deterministic
        - `training`         : If set to false, the samplign will be entirely deterministic
        
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

        -  `masked_pred`   : Prediction based on the current explanation
        -  `original_pred` : Predicion based on the original graph
        -  `edge_mask`     : Current explanaiton
        -  `reg_coefs`     : regularization coefficients

        Return
            loss
        """
        reg_size = self.coeffs["reg_size"]
        reg_ent  = self.coeffs["reg_ent"]

        # Regularization losses
        size_loss = torch.sum(mask) * reg_size
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = reg_ent * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

    def _train(self, indices = None):
        """
        Main method to train the model
        
        Args: 
        - indices: Indices that we want to use for training.
        """
        temp = self.coeffs["temp"]
        sample_bias = self.coeffs["sample_bias"]

        # Make sure the explainer model can be trained
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: temp[0]*((temp[1]/temp[0])**(e/self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.adj).detach()

        # Start training loop
        with tqdm(range(0, self.epochs), desc="[PGExplainer]> ...training") as epochs_bar:
            for e in epochs_bar:
                optimizer.zero_grad()
                loss = torch.FloatTensor([0]).detach()
                t = temp_schedule(e)

                for n in indices:
                    n = int(n)
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
                    sampling_weights = self.explainer_model(input_expl)
                    mask = self._sample_graph(sampling_weights, t, bias=sample_bias).squeeze()

                    masked_pred = self.model_to_explain(feats, graph, edge_weights=mask)
                    original_pred = self.model_to_explain(feats, graph)

                    if self.type == 'node': # we only care for the prediction of the node
                        masked_pred = masked_pred[n].unsqueeze(dim=0)
                        original_pred = original_pred[n]

                    id_loss = self._loss(masked_pred, torch.argmax(original_pred).unsqueeze(0), mask)
                    loss += id_loss

                epochs_bar.set_postfix(loss=f"{loss.item():.4f}")

                loss.backward()
                optimizer.step()


    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.

        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        if indices is None: # Consider all indices
            indices = range(0, self.adj.size(0))

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
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.adj)[1]
            embeds = self.model_to_explain.embedding(self.features, self.adj).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.adj[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph).detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, embeds, index).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()

        expl_graph_weights = torch.zeros(graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights
