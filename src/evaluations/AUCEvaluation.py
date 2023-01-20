import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from .BaseEvaluation import BaseEvaluation



def _eval_AUC_node(explanations, explanation_labels):
    """Evaluate the auc score given explaination and ground truth labels.

    Args
    - `explanations` : predicted labels.
    - `ground_truth` : ground truth labels.
    - `indices` : Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []
    for expl in explanations: # Loop over the explanations for each node
        #print("expl[0]:", expl[0].size()) # edge index
        #print("expl[1]:", expl[1]) # predicted edge labels
        #print("expl labels[0]:", explanation_labels[0].size())
        #print("expl labels[1]:", explanation_labels[1].size())

        ground_truth_node = []
        prediction_node = []

        for i in range(0, expl[0].size(1)): # Loop over all edges in the explanation sub-graph
            prediction_node.append(expl[1][i].item())

            # Graphs are defined bidirectional, so we need to retrieve both edges
            pair = expl[0].T[i] #.numpy()
            pair_rev = torch.Tensor([pair[1], pair[0]])
            idx_edge = np.where((explanation_labels[0].T == pair).all(dim=1))[0]
            idx_edge_rev = np.where((explanation_labels[0].T == pair_rev).all(dim=1))[0]
            #print("idx edge    :", idx_edge)
            #print("idx edge rev:", idx_edge_rev)
            #print("expl edge    :", explanation_labels[1][0].size())
            #print("expl edge rev:", explanation_labels[1][1].size())

            # If any of the edges is in the ground truth set, the edge should be in the explanation
            edge_start = torch.Tensor(idx_edge) in explanation_labels[1][0]
            edge_end = torch.Tensor(idx_edge_rev) in explanation_labels[1][1]
            #print("edge_start:",edge_start,"edge_end:",edge_end)
            #gt = explanation_labels[1][idx_edge] + explanation_labels[1][idx_edge_rev]
            gt = edge_start + edge_end
            #print("ground truth:", gt)
            if gt == 0:
                ground_truth_node.append(0)
            else:
                ground_truth_node.append(1)

            #raise NotImplementedError("sto a debbuggà")

        ground_truth.extend(ground_truth_node)
        predictions.extend(prediction_node)

    score = roc_auc_score(ground_truth, predictions)
    return score


class AUCEvaluation(BaseEvaluation):
    """A class enabling the evaluation of the AUC metric on both graph and node
    classification tasks.
    
    Args
    - `task`(str): either "node" or "graph".
    - `ground_truth` : ground truth labels.
    - `indices` : Which indices to evaluate.
    
    :funcion get_score: obtain the roc auc score.
    """
    def __init__(self, ground_truth, indices, task: str="node"):
        self.task = task
        self.indices = indices
        self.ground_truth = ground_truth

    def get_score(self, explanations):
        """Determines the auc score based on the given list of explanations and
        the list of ground truths
        
        Args
        - `explanations` : list of explanations

        Retruns 
            Area Under Curve (AUC) score for chosen task.
        """
        if self.task == 'graph':
            return NotImplementedError("Graph AUC-score not implemented.")
            return _eval_AUC_graph(explanations, explanation_labels, self.indices)
        elif self.task == 'node':
            return _eval_AUC_node(explanations, self.ground_truth)
