import os
import torch 


rel_path = "/../../datasets/pkls/syn1_adv_GNN.pt"
save_path = os.path.dirname(os.path.realpath(__file__)) + rel_path

# load raw data
with open(save_path, 'rb') as fin:
    data = torch.load(fin)
    print(data.keys())
