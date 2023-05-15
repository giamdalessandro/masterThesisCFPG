import torch
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score

from .BaseEvaluation import BaseEvaluation



def _eval_AUC_node(explanations, explanation_labels, plot_roc: bool=False):
    """Evaluate the auc score given explaination and ground truth labels.

    ### Args
    `explanations` : list
        predicted explanation as a list of (edge_index,edge_labels) tuples, 
        one for each node explained with the edge indices of the node 
        neighborhood and the relative labels;
    
    `explanation_lables` : sparse COO Tensor
        explanation ground truth labels.
    
    ### Returns
    Area Under Curve (AUC) score for node classification.
    """
    ground_truth = []
    predictions = []
    # for easier access densify explanation labels matrix
    expl_labels_dense = explanation_labels[1].to_dense()
    expl_labels_sparse = explanation_labels[1].indices()
    #print("\n\t[DEBUG]> labels:", expl_labels_sparse.size(1))

    visited_edges = torch.zeros(explanation_labels[1].size())
    visited_l = []
    
    with tqdm(explanations, desc="[metrics]> AUC score", disable=False) as eval_step:
        for expl in eval_step: # Loop over each node explanations 
            sub_graph   = expl[0]    # explanation edge-index (i.e. expl. subgraph)
            pred_scores = expl[1]    # explanation edge weights

            ground_truth_node = []
            prediction_node = []
            
            n_edges = sub_graph.size(1)
            for i in range(0, n_edges): # Loop over each edge in the explanation sub-graph
                edge_pred = pred_scores[i].item()
                prediction_node.append(edge_pred)

                # Graphs are defined bidirectional, so we need to retrieve both edges
                # If any of the edges is in the ground truth set, the edge should be in the explanation
                pair = expl[0].T[i].long() #.numpy()          

                edge = expl_labels_dense[pair[0]][pair[1]].item() 
                edge_rev = expl_labels_dense[pair[1]][pair[0]].item() 
                if edge == 1.0 and visited_edges[pair[0]][pair[1]] == 0.0: 
                    visited_edges[pair[0]][pair[1]] = edge_pred
                    visited_l.append(pair)
                #elif edge_rev == 1 and visited_edges[pair[1]][pair[0]] == 0: 
                #    visited_edges[pair[1]][pair[0]] = edge_pred
                #    visited_l.append(pair)

                gt = edge + edge_rev
                #print("ground truth:", gt)
                if gt == 0:
                    ground_truth_node.append(0)
                else:
                    ground_truth_node.append(1)

                #exit("\n[DEBUG]: sto a debbuggà, stacce.")            

            ground_truth.extend(ground_truth_node)
            predictions.extend(prediction_node)

    visited_sparse = visited_edges.to_sparse_coo()
    expl_pred = visited_sparse.values()

    #print("\n\t[DEBUG]> gt:   ", len(ground_truth), "\t#1s:", torch.LongTensor(ground_truth).sum())
    #print("\t[DEBUG]> preds:", len(predictions))

    #missed = (expl_labels_dense - (visited_edges > 0.0).long()).sum()
    #print("\n\t[DEBUG]> expl-edge preds:", expl_pred.size())
    #print("\t[DEBUG]> visited:", len(visited_l))
    #print("\t[DEBUG]> missed: ", missed.item())

    #exit("\n[DEBUG]: sto a debbuggà, stacce.")    

    #fpr, tpr, thres = roc_curve(ground_truth, predictions)
    #print("[eval]> ROC thresholds:", len(thres)) 
    #score = accuracy_score(ground_truth, predictions)
    score = roc_auc_score(ground_truth, predictions)

    if plot_roc:   # plot ROC curve
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

    def get_score(self, explanations, plot: bool=False):
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
            return _eval_AUC_node(explanations, self.ground_truth, plot)
