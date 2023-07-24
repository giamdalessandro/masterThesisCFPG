import os
import subprocess
from tqdm import tqdm

DATASETS = ["syn1","syn2","syn3","syn4"] #
CONVS = ["GAT","GCN"] #,"pGCN"
EXPLAINER = "CFPGv2" # "CFPG", "CFPGv2", "PGEex"
EPOCHS = 100

# returns a list of 8 random small integers between 0 and 255
def get8RandomBytesFromOS():
    """Get random integers (in the interval [0,255]) from OS to be used as seeds."""
    r8 = os.urandom(8)  # official OS entropy source
    byteCodes = list(map(ord, r8.decode('Latin-1')))  # type conversion
    return byteCodes

#ENT_COEFFS = [0.1, 0.5, 1.0, 2.0, 5.0]
#SIZE_COEFFS = [0.1, 0.01, 0.001, 0.0005]
#CF_COEFFS = [0.1, 0.5, 1.0, 2.0, 5.0]
#NUM_HEADS = [3, 5, 8]
SEEDS = get8RandomBytesFromOS()[:4]

params = {
    "syn1" : "--opt Adam --reg-ent 0.5 --reg-size 0.001 --reg-cf 5.0  --heads 5 --hid-gcn 32 --add-att 1.0",
    "syn2" : "--opt Adam --reg-ent 0.5 --reg-size 0.001 --reg-cf 5.0  --heads 5 --hid-gcn 20 --add-att 5.0",
    "syn3" : "--opt Adam --reg-ent 1.0 --reg-size 0.1   --reg-cf 5.0  --heads 5 --hid-gcn 20 --add-att 5.0",
    "syn4" : "--opt Adam --reg-ent 0.5 --reg-size 0.001 --reg-cf 5.0  --heads 5 --hid-gcn 20 --add-att 0.5",
}


script_cmd = "/home/zascerta/virtEnvs/XAI-cuda117/bin/python3 src/explain.py "
rid = 0
#for c in CONVS:
#    for curr in ENT_COEFFS:

#for e in (p_bar := tqdm(DATASETS, desc=f"[multi-run]> experiments", disable=False)):
for s in SEEDS:
    for d in DATASETS:
        script_args = f"-E {EXPLAINER} -D {d} -e {EPOCHS} --conv GAT {params[d]} --seed {s} "
        suffix_args = f"--prefix rParams-GumbelSoftmaxMono-{EPOCHS} --log"
        cmd = script_cmd + script_args + suffix_args

        print("\n\n------------------------------ run id:", rid, f"curr-> {EXPLAINER} - {d} - seed {s}\n")
        returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
        rid += 1

print("\n[runs]> Multi-run DONE...", returned_value)