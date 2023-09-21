from tqdm import tqdm
from numpy import Inf

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD

import torch_geometric
from torch_geometric.utils import k_hop_subgraph 
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, GATv2Conv

from .BaseExplainer import BaseExplainer
from .CFPGv2_em import GCNExplModule, GATExplModule, GCNPerturbExplModule, GAALVExplModule, SMAPExplModule
from utils.graphs import index_edge, create_symm_matrix_from_vec

THRES = 0.1
NODE_BATCH_SIZE = 32

class CFPGv2(BaseExplainer):
    """A class encaptulating CF-PGExplainer v.2 (Counterfactual-PGExplainer v.2)"""
    coeffs = {     ## default values for explainer parameters
        "lr": 0.003,
        "reg_size": 0.5,
        "reg_ent" : 1.0,
        "reg_cf"  : 5.0, 
        "temps": [5.0, 2.0],
        "sample_bias": 0.0,
    }

    def __init__(self, 
            model_to_explain: torch.nn.Module, 
            data_graph: torch_geometric.data.Data,
            conv: str="GCN",
            task: str="node", 
            epochs: int=30, 
            device: str="cpu",
            coeffs: dict=None,
            verbose: bool=False
        ):
        """Initialize CFPGv2 explainer model
        
        #### Args
        model_to_explain : `torch.nn.Module`
            GNN model who's predictions we wish to explain.

        data_graph : `torch_geometric.data.Data`
            the collections of edge_indices representing the graphs.

        task : `str`
            classification task, "node" or "graph".

        epochs : `int`
            amount of epochs to train our explainer.

        coeffs : `dict`
            a dict containing parameters for training the explainer (e.g.
            lr, temprature, etc..).
        """
        super().__init__(model_to_explain, data_graph, task, device)
        self.expl_name = "CFPGv2"
        self.adj = self.data_graph.edge_index.to(device)
        self.features = self.data_graph.x.to(device)
        self.epochs = epochs
        self.conv = conv
        for k,v in coeffs.items():
            self.coeffs[k] = v
        #if verbose: print("\t>> explainer:", self.expl_name)
        if verbose: print("\t>> coeffs:", self.coeffs)
        self.thres = self.coeffs["thres"]
        self.verbose = verbose

        if self.type == "graph": # graph classification model
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3

        # Instantiate the explainer model
        in_feats = self.model_to_explain.embedding_size

        hid_gcn  = self.coeffs["hid_gcn"]
        if conv == "GCN": 
            self.explainer_module = GCNExplModule(in_feats=in_feats, enc_hidden=hid_gcn, # 20
                                        dec_hidden=64, device=device).to(self.device)
        elif conv == "GAT":
            heads    = self.coeffs["heads"] 
            add_att  = self.coeffs["add_att"]
            self.explainer_module = GATExplModule(in_feats=in_feats, enc_hidden=hid_gcn, # 20
                                        dec_hidden=64, heads=heads, add_att=add_att,
                                        device=device).to(self.device)
        elif conv == "pGCN":
            n_nodes = self.features.size(0)
            edges = self.adj
            self.explainer_module = GCNPerturbExplModule(in_feats=in_feats, enc_hidden=hid_gcn,
                                        dec_hidden=64, num_nodes=n_nodes, edges=edges, 
                                        device=device).to(self.device)
        elif conv == "VAE":
            self.explainer_module = GAALVExplModule(in_feats=in_feats, enc_hidden=hid_gcn,
                                        dec_hidden=64, device=device).to(self.device)
            
        elif conv == "SMAP":
            self.explainer_module = SMAPExplModule(in_feats=in_feats, enc_hidden=hid_gcn,
                                        device=device).to(self.device)

            
        self.n_layers = self.explainer_module.n_layers
        self.test_cf_examples = {}
        

    def loss(self, masked_pred: torch.Tensor, original_pred: torch.Tensor, mask: torch.Tensor):
        """Returns the loss score based on the given mask.

        #### Args
        masked_pred : `torch.Tensor`
            Prediction based on the current explanation

        original_pred : `torch.Tensor`
            Predicion based on the original graph

        edge_mask : `torch.Tensor`
            Current explanaiton

        reg_coefs : `torch.Tensor`
            regularization coefficients

        #### Return
            Tuple of Tensors (loss,size_loss,mask_ent_loss,pred_loss)
        """
        reg_size = self.coeffs["reg_size"]
        reg_ent  = self.coeffs["reg_ent"]
        reg_cf   = self.coeffs["reg_cf"]
        EPS = 1e-10  #e-15

        # Size loss
        #mask_mean = mask.mean().detach()

        #m, std = torch.std_mean(mask, unbiased=False)
        #thres = m + std
        #size_loss = (mask > THRES).sum()   # working fine
        #mask = mask.sigmoid()
        size_loss = (mask.sigmoid()).sum()  #
        size_loss = size_loss * reg_size

        # Entropy loss (PGE)
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = reg_ent * torch.mean(mask_ent_reg)
        
        # Explanation loss
        pred_same = (masked_pred.argmax().item() == original_pred).float()
        #cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        nll_loss = -1 * torch.nn.functional.nll_loss(masked_pred, original_pred)
        pred_loss = pred_same * nll_loss * reg_cf

        # ZAVVE: TODO tryin' to optimize objective function for cf case
        loss_total = size_loss + pred_loss + mask_ent_loss #+ kl_loss
        return loss_total, size_loss, mask_ent_loss, pred_loss
    
    def lossVAE(self, 
            masked_pred: torch.Tensor,
            original_pred: torch.Tensor,
            mask: torch.Tensor,
            kl_loss
        ):
        """Returns the VAE loss score based on the given mask.

        #### Args
        masked_pred : `torch.Tensor`
            Prediction based on the current explanation

        original_pred : `torch.Tensor`
            Predicion based on the original graph

        edge_mask : `torch.Tensor`
            Current explanaiton

        reg_coefs : `torch.Tensor`
            regularization coefficients

        #### Return
            Tuple of Tensors (loss,size_loss,mask_ent_loss,pred_loss)
        """
        reg_size = self.coeffs["reg_size"]
        reg_ent  = self.coeffs["reg_ent"]
        reg_cf   = self.coeffs["reg_cf"]
        EPS = 1e-15

        # Size loss
        size_loss = -mask.sum()
        size_loss = size_loss * reg_size

        # Entropy loss (PGE)
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = reg_ent * torch.mean(mask_ent_reg)

        # Misclassifiaction loss
        pred_same = (masked_pred.argmax().item() == original_pred).float()
        nll_loss = -1 * torch.nn.functional.nll_loss(masked_pred, original_pred)
        pred_loss = pred_same * nll_loss * reg_cf

        # ZAVVE: TODO tryin' to optimize objective function for cf case
        loss_total = size_loss + pred_loss + mask_ent_loss + (0.0*kl_loss)  
        return loss_total, size_loss, mask_ent_loss, pred_loss

    def _train(self, indices=None):
        """Main method to train the modeledge_weights=None
        
        #### Args 
        indices : `torch.Tensor`
            Indices that we want to use for training.
        """
        lr = self.coeffs["lr"]
        temp = self.coeffs["temps"]
        sample_bias = self.coeffs["sample_bias"]
        early_stop = self.coeffs["early_stop"]

        # Make sure the explainer model can be trained
        self.explainer_module.train()

        # Create optimizer and temperature schedule
        if self.coeffs["opt"] == "Adam": 
            optimizer = Adam(self.explainer_module.parameters(), lr=lr)
        elif self.coeffs["opt"] == "SGD":
            optimizer = SGD(self.explainer_module.parameters(), lr=lr)
        elif self.coeffs["opt"] == "SGDm":
            optimizer = SGD(self.explainer_module.parameters(), lr=lr, nesterov=True, momentum=0.9)

        temp_schedule = lambda e: temp[0]*((temp[1]/temp[0])**(e/self.epochs))

        ## If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.adj)[0].detach().to(self.device)

        # use NeighborLoader to sample batch_size nodes and their respective 3-hop neighborhood
        gnn_iter = 3               # GCN model has 3 mp iteration
        num_neighbors = -1         # -1 for all neighbors
        loader = NeighborLoader(
            self.data_graph,
            num_neighbors=[num_neighbors] * gnn_iter,          
            batch_size=NODE_BATCH_SIZE, 
            input_nodes=indices.cpu(),
            disjoint=False,
        )
        n_indices = indices.size(0)
        n_batches = len(loader)

        self.history = {}
        epoch_loss_tot  = []
        epoch_loss_size = []
        epoch_loss_ent  = []
        epoch_loss_pred = []
        epoch_cf_ex     = []

        self.cf_examples = {}
        best_loss = Inf
        #self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        # Start training loop
        for e in (p_bar := tqdm(range(0, self.epochs), desc=f"[{self.expl_name}]> training", disable=not(self.verbose))):
            optimizer.zero_grad()
            loss_total = torch.FloatTensor([0]).detach().to(self.device)
            size_total = torch.FloatTensor([0]).detach().to(self.device)
            ent_total  = torch.FloatTensor([0]).detach().to(self.device)
            pred_total = torch.FloatTensor([0]).detach().to(self.device)
            t = temp_schedule(e)
            
            b_id = 0
            for node_batch in loader:
                if b_id == (n_batches-1):   # last batch may be smaller
                    curr_batch_size = n_indices % NODE_BATCH_SIZE
                else:
                    curr_batch_size = NODE_BATCH_SIZE

                #if self.type == 'node':
                batch_feats  = node_batch.x
                batch_graph  = node_batch.edge_index
                global_n_ids = node_batch.n_id
                b_id += 1

                for b_n_idx in range(curr_batch_size):
                    # only need node_id neighnbors to compute the explainer input
                    global_idx = global_n_ids[b_n_idx].to(self.device)

                    # we only consider 3hop-neighborhood for explaining
                    sub_nodes, sub_index, n_map, _ = k_hop_subgraph(b_n_idx, 3, batch_graph, relabel_nodes=True)
                    sub_index = sub_index.to(self.device)
                    sub_feats = batch_feats[sub_nodes, :].to(self.device)

                    # relabel sub-graph with global node indices   # still needed?
                    global_n_ids = global_n_ids.to(self.device)
                    sub_graph = torch.take(global_n_ids,sub_index)    
                    

                    # compute explanation mask
                    expl_feats = embeds[global_n_ids].to(self.device)
                    #mask = self.explainer_module(expl_feats, sub_index, n_map, temp=t, bias=sample_bias)
                    mask = self.explainer_module(embeds, batch_graph, n_map, temp=t, bias=sample_bias)
                    #print("\n\t>> node id:", global_idx)                        
                    #print("\t>> mask mean:", mask.mean())
                    #print("\t>> over mean:", (mask > 0.1).sum())

                    ## basic minus thresholds
                    #cf_adj = torch.ones(mask.size()).to(self.device) 
                    #cf_mask = torch.nn.functional.gumbel_softmax(mask, tau=t, hard=True, dim=0)
                    cf_mask = (1 - mask) #.abs()
                    #cf_mask = (mask <= THRES).float()

                    ## top-k thresholds
                    #_, sorted_index = torch.sort(mask.squeeze(), descending=True)
                    #top_k = sorted_index[:6] #12
                    #cf_mask = torch.zeros(mask.size())# - 0.5
                    #cf_mask[top_k] = 1.0

                    ## mean thresholds
                    #m, std = torch.std_mean(mask, unbiased=False)
                    #thres = m + std
                    #cf_mask = (mask <= thres).float() #mask.mean()
                    #cf_mask = (mask.mean() - mask*2).abs()

                    ## softmax thresholds
                    #cf_mask = mask.argmin(dim=1).float()
                    #mask = mask[:,1]

                    #masked_pred, cf_feat = self.model_to_explain(sub_feats, sub_index, edge_weights=cf_mask, cf_expl=True)
                    masked_pred, cf_feat = self.model_to_explain(batch_feats, batch_graph, edge_weights=cf_mask, cf_expl=True)
                    #original_pred = self.model_to_explain(sub_feats, sub_index)
                    original_pred = self.model_to_explain(batch_feats, batch_graph)


                    sub_node_idx = n_map.item()
                    if self.type == 'node': # node class prediction
                        # when considering the features subset, node prediction is at index 0
                        masked_pred = masked_pred[sub_node_idx]
                        op = original_pred[sub_node_idx]
                        original_pred = op.argmax()
                        pred_same = (masked_pred.argmax() == original_pred)

                    if self.conv == "VAE":
                        id_loss, size_loss, ent_loss, pred_loss = self.lossVAE(masked_pred=masked_pred, 
                                                                original_pred=original_pred, 
                                                                mask=mask,
                                                                kl_loss=self.explainer_module.kl)
                    else:
                        id_loss, size_loss, ent_loss, pred_loss = self.loss(masked_pred=masked_pred, 
                                                                original_pred=original_pred, 
                                                                mask=mask)

                    loss_total += id_loss
                    size_total += size_loss
                    ent_total  += ent_loss
                    pred_total += pred_loss

                    # if masked prediction is different from original, save the CF example
                    if pred_same == 0:
                        #print("cf example found for node", global_idx)
                        #if id_loss < best_loss: best_loss = id_loss
                        cf_ex = {"loss": id_loss, "mask": cf_mask, "feats": cf_feat[sub_node_idx]}
                        try: 
                            # update found CF only if loss has improved
                            if id_loss < self.cf_examples[str(global_idx)]["loss"]:
                                self.cf_examples[str(global_idx)] = cf_ex
                        except KeyError:
                            self.cf_examples[str(global_idx)] = cf_ex

            p_bar.set_postfix(loss=f"{loss_total.item():.4f}", l_size=f"{size_total.item():.4f}",
                                    l_ent=f"{ent_total.item():.4f}", l_pred=f"{pred_total.item():.4f}")
                    
            # metrics to plot
            epoch_loss_tot.append(loss_total.item())
            epoch_loss_size.append(size_total.item())
            epoch_loss_ent.append(ent_total.item())
            epoch_loss_pred.append(pred_total.item())
            epoch_cf_ex.append(len(self.cf_examples.keys()))

            loss_total.backward()
            optimizer.step()

            if loss_total < best_loss: 
                best_loss = loss_total
                early_stop = self.coeffs["early_stop"]
            elif e >= 14: 
                early_stop -= 1
                if early_stop == 0:
                    print(f"\n\t>> No loss improvements for {self.coeffs['early_stop']} epochs... Early STOP") 
                    break


        self.history["train_loss"] = {
            "loss_tot"  : epoch_loss_tot,
            "loss_size" : epoch_loss_size,
            "loss_ent"  : epoch_loss_ent,
            "loss_pred" : epoch_loss_pred,
        }
        self.history["cf_fnd"] = epoch_cf_ex
        self.history["cf_tot"] = n_indices


    def prepare(self, indices=None):
        """Prepars the explanation method for explaining. When using a parametrized 
        explainer like PGExplainer, we first need to train the explainer MLP.

        #### Args
        indices : `list`
            node indices over which we wish to train.
        """
        if indices is None: # Consider all indices
            indices = range(0, self.adj.size(0))

        indices = torch.LongTensor(indices).to(self.device)   
        self._train(indices=indices)

    def _extract_cf_example(self, index, sub_graph, cf_mask):
        """Given the computed CF edge mask for a node prediction extracts
        the related CF example, if any."""
        with torch.no_grad():
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

    def explain(self, index: int):
        """Given a node index returns its explanation subgraph. 
        This only gives sensible results if the prepare method has already been called.

        #### Args
        index : `int`
            index of the node to be explained

        #### Return
            The tuple (explanation subgraph, edge weights)
        """
        index = int(index)
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            sub_nodes, sub_graph, n_map, _ = k_hop_subgraph(index, 3, self.adj, relabel_nodes=False)
            embeds = self.model_to_explain.embedding(self.features, self.adj)[0].detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.adj[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph)[0].detach()

        # Use explainer mlp to get an explanation
        #expl_feats = embeds[sub_nodes, :].to(self.device)
        #mask = self.explainer_module(sub_feats, sub_graph, n_map)
        mask = self.explainer_module(embeds, sub_graph, index, train=False)
        #mask = torch.nn.functional.gumbel_softmax(mask, tau=1.0, hard=False, dim=0)

        ## to get opposite of cf-mask, i.e. explanation
        #cf_mask = (1 - mask)
        cf_mask = (mask <= THRES).float()

        ## top-k thresholds
        #_, sorted_index = torch.sort(mask.squeeze(), descending=True)
        #top_k = sorted_index[:6] #12
        #cf_mask = torch.zeros(mask.size())# - 0.5
        #cf_mask[top_k] = 1.0
        
        ## mean thresholds
        #m, std = torch.std_mean(mask, unbiased=False)
        #thres = m + std
        #cf_mask = (mask <= thres).float()  #mask.mean()
        #cf_mask = (mask.mean() - mask*2)
        
        ## softmax thresholds
        #cf_mask = mask.argmin(dim=1).float()
        #cf_mask = (mask <= THRES).float()
        #print("\n\t>> cf_mask:", cf_mask)
        #print("\t>> cf_mask:", cf_mask.sum())
        #print("\t>> argmax:", mask.argmax(dim=1).sum())
        #mask = mask[:,1]
        #print("\t>> mask:", mask)
        #print("\t>> over:", (mask > 0.5).sum())
        #exit(0)

        self._extract_cf_example(index, sub_graph, cf_mask)

        expl_graph_weights = torch.zeros(sub_graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = sub_graph.T[i]
            t = index_edge(sub_graph, pair)
            expl_graph_weights[t] = mask[i]

        return sub_graph, expl_graph_weights

