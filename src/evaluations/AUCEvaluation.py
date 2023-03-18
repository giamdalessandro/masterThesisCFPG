import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from .BaseEvaluation import BaseEvaluation



def _eval_AUC_node(explanations, explanation_labels, plot_roc: bool=True):
    """Evaluate the auc score given explaination and ground truth labels.

    Args
    - `explanations`(list): predicted labels, list of tuples (edge,edge_labels);
    - `explanation_lables`(sparse COO Tensor): explanation ground truth labels.
    - `indices` : Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []
    # for easier access densify explanation labels matrix
    expl_lab_dense = explanation_labels[1].to_dense()
    #pred_scores = explanations[1]
    #print("[AUC]> pred scores:", (pred_scores > 0.5).sum())

    with tqdm(explanations, desc="[metrics]> AUC score", disable=False) as eval_step:
        for expl in eval_step: # Loop over the explanations for each node
            #print("[AUC]> expl graph (expl[0]):", expl[0].size())    # edge index
            pred_scores = expl[1]
            pred_mean = expl[1].mean()
            #print("[AUC]> expl preds (expl[1]):", (expl[1] > pred_mean.item()).sum())    # predicted edge labels
            #print("[AUC]> gt edges :", explanation_labels[0].size())
            #print("[AUC]> gt labels:", explanation_labels[1].size())

            ground_truth_node = []
            prediction_node = []
            
            for i in range(0, expl[0].size(1)): # Loop over all edges in the explanation sub-graph
                prediction_node.append(expl[1][i].item())
                #print(f"[AUC] prediction: {expl[1][i].item():.4f}")

                # Graphs are defined bidirectional, so we need to retrieve both edges
                pair = expl[0].T[i].long() #.numpy()
                #pair_rev = torch.LongTensor([pair[1], pair[0]])
                #idx_edge = np.where((explanation_labels[0].T == pair).all(dim=1))[0]
                #idx_edge_rev = np.where((explanation_labels[0].T == pair_rev).all(dim=1))[0]
                #print("\nidx edge     :", idx_edge)
                #print("idx edge rev :", idx_edge_rev)
                #print("expl edge    :", explanation_labels[1].T.size())

                # If any of the edges is in the ground truth set, the edge should be in the explanation
                edge     = expl_lab_dense[pair[0]][pair[1]].item() 
                edge_rev = expl_lab_dense[pair[1]][pair[0]].item() 
                gt = edge + edge_rev
                #print("ground truth:", gt)
                if gt == 0:
                    ground_truth_node.append(0)
                else:
                    ground_truth_node.append(1)

                #exit("\n[DEBUG]: sto a debbuggà, stacce.")            

            ground_truth.extend(ground_truth_node)
            predictions.extend(prediction_node)

    score = roc_auc_score(ground_truth, predictions)

    if plot_roc:
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay

        RocCurveDisplay.from_predictions(
            ground_truth,
            predictions,
            name=f"expl.edges vs the rest",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("FP rate")
        plt.ylabel("TP rate")
        plt.title("ROC curve")
        plt.legend()
        plt.show()

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
