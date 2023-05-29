import subprocess

script_cmd  = "/home/zascerta/virtEnvs/XAI-cuda117/bin/python3 src/explain.py "

DATASETS = ["syn1","syn2","syn3","syn4"]
CONVS = ["GCN","GAT"]

ENT_COEFFS = [0.1, 0.01, 0.001]
fixed_ent = 1.0

SIZE_COEFFS = [0.1, 0.01, 0.001]
fixed_size = 0.01

CF_COEFFS = [1.0, 2.0, 5.0]
fixed_cf  = 1.0


SET_0 = [ENT_COEFFS,fixed_size,fixed_cf]
SET_1 = [fixed_ent,SIZE_COEFFS,fixed_cf]
SET_2 = [fixed_ent,fixed_size,CF_COEFFS]


rid = 0
for c in CONVS:
    for d in DATASETS:
        for curr in SIZE_COEFFS:
            script_args = f"-E CFPGv2 -D {d} -e 10 --conv {c} --reg-ent {fixed_ent} --reg-size {curr} --reg-cf {fixed_cf} "
            suffix_args = f"--prefix rSIZE-tests --log"
            cmd = script_cmd + script_args + suffix_args

            print("\n\n------------------------------ run id:", rid, f"curr coeff {curr}\n")
            returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
            rid += 1

print("\n[runs]> Multi-run DONE...", returned_value)