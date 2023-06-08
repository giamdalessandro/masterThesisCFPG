import subprocess
import  functools
import  os


DATASETS = ["syn4"] #["syn1","syn2","syn3","syn4"]
CONVS = ["pGCN","GCN","GAT"]
EXPLAINER = "PGEex"
EPOCHS = 20

# returns a list of 8 random small integers between 0 and 255
def get8RandomBytesFromOS():
    """Get random integers (in the interval [0,255]) from OS to be used as seeds."""
    r8 = os.urandom(8)  # official OS entropy source
    byteCodes = list(map(ord, r8.decode('Latin-1')))  # type conversion
    return byteCodes


ENT_COEFFS = [0.1, 0.5, 1.0, 2.0]
#SIZE_COEFFS = [0.1, 0.01, 0.001]
#CF_COEFFS = [0.5, 1.0, 2.0, 5.0]
NUM_HEADS = [3, 5, 8]


SEEDS = get8RandomBytesFromOS()

script_cmd  = "/home/zascerta/virtEnvs/XAI-cuda117/bin/python3 src/explain.py "
rid = 0
for s in SEEDS:
    #for h in NUM_HEADS:
    for d in DATASETS:
        #for curr in ENT_COEFFS:
        script_args = f"-E {EXPLAINER} -D {d} -e {EPOCHS} --seed {s} "
        suffix_args = f"--prefix replicationPGEex-rBest-{EPOCHS}-tests --log"
        cmd = script_cmd + script_args + suffix_args

        print("\n\n------------------------------ run id:", rid, f"curr-> {EXPLAINER} - {d} - seed {s}\n")
        returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
        rid += 1

print("\n[runs]> Multi-run DONE...", returned_value)