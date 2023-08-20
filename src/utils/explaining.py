from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

from explainers.PGExplainer import PGExplainer
from explainers.CFPGExplainer import CFPGExplainer
from explainers.CFPGv2 import CFPGv2
from explainers.OneHopExplainer import OneHopExplainer, PerfectExplainer
#from explainers.PCFExplainer import PCFExplainer
#from utils.graphs import normalize_adj


def explainer_selector(cfg, model, graph, s_args, verbose: bool=False):
    """Select explainer model for the given params"""
    expl = s_args.explainer
    epochs = s_args.epochs
    device = s_args.device
    if verbose: print(Fore.MAGENTA+"\n[explain]> Loading",f"{expl}",Fore.MAGENTA+"explainer..")

    cfg["expl_params"]["reg_ent"] = cfg["expl_params"]["reg_ent"] if s_args.reg_ent == 0.0 else s_args.reg_ent
    cfg["expl_params"]["reg_size"] = cfg["expl_params"]["reg_size"] if s_args.reg_size == 0.0 else s_args.reg_size

    if expl == "PGEex":
        explainer = PGExplainer(model, graph, epochs=epochs, device=device, coeffs=cfg["expl_params"]) # needs 'GNN' model
    else:
        cfg["expl_params"]["opt"] = cfg["expl_params"]["opt"] if s_args.opt == "base" else s_args.opt
        cfg["expl_params"]["reg_cf"] = cfg["expl_params"]["reg_cf"] if s_args.reg_cf == 0.0 else s_args.reg_cf

        if expl == "CFPG":
            explainer = CFPGExplainer(model, graph, epochs=epochs, device=device, coeffs=cfg["expl_params"])
        elif expl == "CFPGv2":
            conv = cfg["expl_params"]["conv"] if s_args.conv == "base" else s_args.conv
            cfg["expl_params"]["conv"]    = conv
            cfg["expl_params"]["heads"]   = s_args.heads
            cfg["expl_params"]["add_att"] = s_args.add_att
            cfg["expl_params"]["hid_gcn"] = s_args.hid_gcn
            explainer = CFPGv2(model, graph, conv=conv, epochs=epochs, coeffs=cfg["expl_params"], verbose=verbose)
            
        # baseline explainers    
        elif expl == "1hop":
            explainer = OneHopExplainer(model, graph, device=device)
        elif expl == "perfEx":
            explainer = PerfectExplainer(model, graph, device=device)

    return explainer




#### to load CF-GNN model
#if EXPLAINER == "CF-GNN":
#    # need dense-normalized adjacency matrix for GCNSynthetic model
#    v = torch.ones(edge_index.size(1))
#    s = (graph.num_nodes,graph.num_nodes)
#    dense_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
#    norm_adj = normalize_adj(dense_index)

#elif EXPLAINER == "CF-GNN":
#   explainer = PCFExplainer(model, graph, norm_adj, epochs=epochs, device=device, coeffs=cfg["expl_params"])