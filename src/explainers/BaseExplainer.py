from abc import ABC, abstractmethod
import torch_geometric


class BaseExplainer(ABC):
    def __init__(self, model_to_explain, data_graph: torch_geometric.data.Data, task: str, device: str="cpu"):
        self.model_to_explain = model_to_explain.to(device)
        self.data_graph = data_graph
        self.type   = task
        self.device = device

    @abstractmethod
    def prepare(self, args):
        """Prepars the explanation method for explaining.
        Can for example be used to train the method"""
        pass

    @abstractmethod
    def explain(self, index):
        """
        Main method for explaining samples
        :param index: index of node/graph in self.graphs
        :return: explanation for sample
        """
        pass

