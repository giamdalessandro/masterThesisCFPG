import torch
import torch_geometric

from utils.graphs import index_edge

from .BaseExplainer import BaseExplainer


class OneHopExplainer(BaseExplainer):
    """Class implementing a counterfactual baseline, considering the case
    when the CF perturbation is the 1hop-neighborhood of a node."""
    def __init__(self, 
            model_to_explain, 
            data_graph: torch_geometric.data.Data, 
            task: str="node", 
            device: str="cpu"
        ):
        super().__init__(model_to_explain, data_graph, task, device)
        self.expl_name = "OneHopExpl"
        self.features = self.data_graph.x.to(self.device)
        self.adj = self.data_graph.edge_index.to(self.device)
        print("\t>> explainer:", self.expl_name)
        
    def _get_1hop_mask(self, index, sub_graph):
        """Get 1hop-neighbors of node `index` as the explanation mask."""
        n_rows = self.features.size(0)   # number of nodes
        mask = torch.zeros((n_rows,n_rows)) + 0.001
        #mask = matrix.new_full(matrix.size(), 0.001)
        
        one_hop = torch_geometric.utils.k_hop_subgraph(index, 1, self.adj)[1]
        mask[one_hop[0],one_hop[1]] = 0.999
        #mask[one_hop[1],one_hop[0]] = 0.999  # graph is undirected

        mask = mask[sub_graph[0],sub_graph[1]].to_sparse_coo()        
        return mask.values()
    
    def _extract_cf_example(self, index, sub_graph, cf_mask):
        """Given the computed CF edge mask for a node prediction extracts
        the related CF example, if any."""
        masked_pred, cf_feat = self.model_to_explain(self.features, sub_graph, edge_weights=cf_mask, cf_expl=True)
        original_pred = self.model_to_explain(self.features, sub_graph)
        
        masked_pred   = masked_pred[index]
        original_pred = original_pred[index].argmax()
        pred_same = (masked_pred.argmax() == original_pred)

        if not pred_same:
            cf_ex = {"mask": cf_mask, "feats": cf_feat[index]}
            try: 
                self.test_cf_examples[str(index)] = cf_ex
            except KeyError:
                self.test_cf_examples[str(index)] = cf_ex
        return


    def prepare(self, indices=None):
        """Prepars the explanation method for explaining."""
        self.test_cf_examples = {}
        print(f"[{self.expl_name}]> Getting 1hop-neighbors as explanation, no training needed.")
        return

    def explain(self, index):
        """Given the index of a node/graph this method returns its explanation. 
        This only gives sensible results if the prepare method has already been called.

        #### Args
        index : `int`
            index of the node/graph that we wish to explain

        #### Return
            explanaiton graph and edge weights
        """
        index = int(index)
        # Similar to the original paper we only consider a subgraph for explaining
        sub_graph = torch_geometric.utils.k_hop_subgraph(index, 3, self.adj)[1]
        embeds = self.model_to_explain.embedding(self.features, self.adj)[0].detach()

        # Use 1hop-neighborhood as explanation
        mask = self._get_1hop_mask(index,sub_graph)
        cf_mask = (1 - mask).abs()

        self._extract_cf_example(index, sub_graph, cf_mask)

        expl_graph_weights = torch.zeros(sub_graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = sub_graph.T[i]
            t = index_edge(sub_graph, pair)
            expl_graph_weights[t] = mask[i]

        return sub_graph, expl_graph_weights
    


class PerfectExplainer(BaseExplainer):
    """Class implementing a counterfactual baseline, considering the cases when 
    the counterfactual perturbation coincide exactly with the exaplanation."""
    def __init__(self, 
            model_to_explain, 
            data_graph: torch_geometric.data.Data,
            task: str="node", 
            device: str="cpu"
        ):
        super().__init__(model_to_explain, data_graph, task, device)
        self.expl_name = "PerfectExpl"
        self.features = self.data_graph.x.to(self.device)
        self.adj = self.data_graph.edge_index.to(self.device)
        self.expl_labels = self.data_graph.edge_label.to(self.device)
        print("\t>> explainer:", self.expl_name)
    
    def _get_expl_mask(self, index, sub_graph):
        """Get explanation ground truth of a node `index` as the explanation mask."""
        n_rows = self.features.size(0)   # number of nodes
        mask = torch.zeros((n_rows,n_rows)) + 0.001

        # labels are a list of edges in the form: e -> [i,j]
        pn_labels = self.data_graph.pn_labels["per_node_labels"][str(index)]

        expl_edges = torch.LongTensor(pn_labels).T  # use sparse repr for better indexing
        mask[expl_edges[0],expl_edges[1]] = 0.999
        mask[expl_edges[1],expl_edges[0]] = 0.999  # graph is undirected

        #self.correct_labels[expl_edges[0],expl_edges[1]] = 1.0

        mask = mask[sub_graph[0],sub_graph[1]].to_sparse_coo()
        return mask.values()
    
    def _extract_cf_example(self, index, sub_graph, cf_mask):
        """Given the computed CF edge mask for a node prediction extracts
        the related CF example, if any."""
        masked_pred, cf_feat = self.model_to_explain(self.features, sub_graph, edge_weights=cf_mask, cf_expl=True)
        original_pred = self.model_to_explain(self.features, sub_graph)
        
        masked_pred   = masked_pred[index]
        original_pred = original_pred[index].argmax()
        pred_same = (masked_pred.argmax() == original_pred)

        if not pred_same:
            cf_ex = {"mask": cf_mask, "feats": cf_feat[index]}
            try: 
                self.test_cf_examples[str(index)] = cf_ex
            except KeyError:
                self.test_cf_examples[str(index)] = cf_ex
        return


    def prepare(self, indices=None):
        """Prepars the explanation method for explaining."""
        #self.correct_labels = torch.zeros((self.features.size(0),self.features.size(0))).to(self.device)
        self.labeled_nodes = {}

        self.test_cf_examples = {}
        print(f"[{self.expl_name}]> Getting ground truth explanation to check CF performances.")
        return

    def explain(self, index):
        """Given the index of a node/graph this method returns its explanation. 
        This only gives sensible results if the prepare method has already been called.

        #### Args
        index : `int` 
            index of the node/graph that we wish to explain

        #### Return
            explanaiton graph and edge weights
        """
        index = int(index)
        # Similar to the original paper we only consider a subgraph for explaining
        sub_graph = torch_geometric.utils.k_hop_subgraph(index, 3, self.adj)[1]
        embeds = self.model_to_explain.embedding(self.features, self.adj)[0].detach()

        # Use 1hop-neighborhood as explanation
        mask = self._get_expl_mask(index,sub_graph)
        cf_mask = (1 - mask).abs()

        self._extract_cf_example(index, sub_graph, cf_mask)

        expl_graph_weights = torch.zeros(sub_graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = sub_graph.T[i]
            t = index_edge(sub_graph, pair)
            expl_graph_weights[t] = mask[i]

        return sub_graph, expl_graph_weights
    
