import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from .BaseEvaluation import BaseEvaluation


def _eval_AUC(task, explanations, explanation_labels, indices):
    """Determines based on the task which auc evaluation method should be called to determine the AUC score

    :param task: str either "node" or "graph".
    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    if task == 'graph':
        return NotImplementedError("Grpah classification not implemented.")
        return _eval_AUC_graph(explanations, explanation_labels, indices)
    elif task == 'node':
        return _eval_AUC_node(explanations, explanation_labels)

def _eval_AUC_node(explanations, explanation_labels):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []
    for expl in explanations: # Loop over the explanations for each node

        ground_truth_node = []
        prediction_node = []

        for i in range(0, expl[0].size(1)): # Loop over all edges in the explanation sub-graph
            prediction_node.append(expl[1][i].item())

            # Graphs are defined bidirectional, so we need to retrieve both edges
            pair = expl[0].T[i] #.numpy()
            pair_rev = torch.Tensor([pair[1], pair[0]])
            idx_edge = np.where((explanation_labels[0].T == pair).all(dim=1))[0]
            idx_edge_rev = np.where((explanation_labels[0].T == pair_rev).all(dim=1))[0]
            print("idx edge    :", idx_edge)
            print("idx edge rev:", idx_edge_rev)
            print("expl labels  :", explanation_labels[1].size())
            #print("expl edge    :", explanation_labels[1][idx_edge])
            #print("expl edge rev:", explanation_labels[1][idx_edge_rev])

            # If any of the edges is in the ground truth set, the edge should be in the explanation
            gt = explanation_labels[1][idx_edge] + explanation_labels[1][idx_edge_rev]
            print("ground truth:", gt)
            if gt == 0:
                ground_truth_node.append(0)
            else:
                ground_truth_node.append(1)

        ground_truth.extend(ground_truth_node)
        predictions.extend(prediction_node)

    score = roc_auc_score(ground_truth, predictions)
    return score


class AUCEvaluation(BaseEvaluation):
    """
    A class enabling the evaluation of the AUC metric on both graphs and nodes.
    
    :param task: str either "node" or "graph".
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate.
    
    :funcion get_score: obtain the roc auc score.
    """
    def __init__(self, ground_truth, indices, task: str="node"):
        self.task = task
        self.indices = indices

        #labels = []
        #print("gt.T :", ground_truth[0])
        #for pair in ground_truth[0]:
        #    labels.append(ground_truth[1][pair[0],pair[1]])
        self.ground_truth = ground_truth

    def get_score(self, explanations):
        """
        Determines the auc score based on the given list of explanations and the list of ground truths
        :param explanations: list of explanations
        :return: auc score
        """
        return _eval_AUC(self.task, explanations, self.ground_truth, self.indices)
