import subprocess
import os


DATASETS = ["syn1","syn2","syn3","syn4"]#
CONVS = ["GCN","GAT"] # "pGCN"
EXPLAINER = "CFPGv2" # "CFPG", "CFPGv2", "PGEex"
EPOCHS = 20

# returns a list of 8 random small integers between 0 and 255
def get8RandomBytesFromOS():
    """Get random integers (in the interval [0,255]) from OS to be used as seeds."""
    r8 = os.urandom(8)  # official OS entropy source
    byteCodes = list(map(ord, r8.decode('Latin-1')))  # type conversion
    return byteCodes

#ENT_COEFFS = [0.1, 0.5, 1.0, 2.0]
#SIZE_COEFFS = [0.1, 0.01, 0.001]
#CF_COEFFS = [0.5, 1.0, 2.0, 5.0, 10]
#NUM_HEADS = [3, 5, 8]
SEEDS = get8RandomBytesFromOS()[:4]


script_cmd = "/home/zascerta/virtEnvs/XAI-cuda117/bin/python3 src/explain.py "
rid = 0
#for c in CONVS:
for h in [32,50]:
    for s in SEEDS:
        for d in DATASETS:
            script_args = f"-E {EXPLAINER} -D {d} -e {EPOCHS} --hig-gcn {h} --seed {s} "
            suffix_args = f"--prefix repl{EXPLAINER}-ariStatic2-Test-{EPOCHS} --log"
            cmd = script_cmd + script_args + suffix_args

            print("\n\n------------------------------ run id:", rid, f"curr-> {EXPLAINER} - {d} - seed {s}\n")
            returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
            rid += 1

print("\n[runs]> Multi-run DONE...", returned_value)