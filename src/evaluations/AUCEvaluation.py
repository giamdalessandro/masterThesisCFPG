import torch
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score

from .BaseEvaluation import BaseEvaluation



def _eval_AUC_node(explanations, explanation_labels):
    """Evaluate the auc score given explaination and ground truth labels.

    ### Args
    explanations : `list`
        predicted explanation as a list of (edge_index,edge_labels) tuples, 
        one for each node explained with the edge indices of the node 
        neighborhood and the relative labels;
    
    explanation_lables : sparse COO Tensor
        explanation ground truth labels.
    
    ### Returns
    Area Under Curve (AUC) score for node classification.
    """
    ground_truth = []
    predictions = []
    # for easier access densify explanation labels matrix
    expl_labels_dense = explanation_labels[1].to_dense()
    #expl_labels_sparse = explanation_labels[1].indices()
    #print("\n\t[DEBUG]> labels:", expl_labels_sparse.size(1))
    #print("\t[DEBUG]> expl:", explanations[0][0].size())
    #print("\t[DEBUG]> expl:", explanations[0][1].size())

    # Need per-node labels to avoid overlapping between explanation
    # edges of different nodes that may be in the same 3-hop-neighborhood,
    # i.e. those explanation edges that are in the 3-hop-neighborhood of a
    # node even though they belong to some other node explanation. 
    expl_pn_labels = explanation_labels[2]["per_node_labels"]
    
    for expl in (t := tqdm(explanations, desc="[metrics]> AUC score", disable=True, colour="magenta")):
        #for expl in eval_step: # Loop over each node explanations 
        sub_graph   = expl[0]    # explanation edge-index (i.e. expl. subgraph)
        pred_scores = expl[1]    # explanation edge weights
        node_idx    = expl[2]

        ground_truth_node = []
        prediction_node = []
        
        # node explanation labels as a matrix
        mask = torch.zeros(expl_labels_dense.size())
        expl_edges = torch.LongTensor(expl_pn_labels[str(node_idx)]).T  # use sparse repr for better indexing
        mask[expl_edges[0],expl_edges[1]] = 1
        mask[expl_edges[1],expl_edges[0]] = 1  # graph is undirected

        n_edges = sub_graph.size(1)
        for i in range(0, n_edges): # Loop over each edge in the explanation sub-graph
            edge_pred = pred_scores[i].item()
            prediction_node.append(edge_pred)

            # Graphs are defined bidirectional, so we need to retrieve both edges
            # If any of the edges is in the ground truth set, the edge should be in the explanation
            #edge     = expl_labels_dense[pair[0]][pair[1]].item()  # to use old labels
            #edge_rev = expl_labels_dense[pair[1]][pair[0]].item()  # to use old labels
            pair = sub_graph.T[i].long() #.numpy()          
            edge = mask[pair[0]][pair[1]]
            edge_rev = mask[pair[1]][pair[0]]
            
            gt = edge + edge_rev
            #print("ground truth:", gt)
            if gt == 0:
                ground_truth_node.append(0)
            else:
                ground_truth_node.append(1)

        #exit("\n[DEBUG]: sto a debbuggà, stacce.")            
        ground_truth.extend(ground_truth_node)
        predictions.extend(prediction_node)

    #fpr, tpr, thres = roc_curve(ground_truth, predictions)
    #print("[eval]> ROC thresholds:", len(thres)) 
    #score = accuracy_score(ground_truth, predictions)
    score = roc_auc_score(ground_truth, predictions)

    return score, ground_truth, predictions


class AUCEvaluation(BaseEvaluation):
    """A class enabling the evaluation of the AUC metric on both graph and node
    classification tasks.
    
    :funcion `get_score`: obtain the roc auc score.
    """
    def __init__(self, ground_truth, indices, task: str="node"):
        """#### Args
        task : `str` 
            either "node" or "graph"

        ground_truth : `list` 
            ground truth labels, a list of (graph,edge_labels) tuples
        
        indices : `list`
            Which indices to evaluate
        """
        self.task = task
        self.indices = indices
        self.ground_truth = ground_truth

    def get_score(self, explanations):
        """Determines the auc score based on the given list of explanations and
        the list of ground truths
        
        #### Args
        explanations : `list` 
            list of explanations

        #### Retruns 
            Area Under Curve (AUC) score for chosen task.
        """
        if self.task == 'graph':
            return NotImplementedError("Graph AUC-score not implemented.")
            return _eval_AUC_graph(explanations, explanation_labels, self.indices)
        elif self.task == 'node':
            return _eval_AUC_node(explanations, self.ground_truth)
