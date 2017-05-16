import numpy as np
import math

data_path = "../data/sentences_test"
perp_path = "../final_results/group02.perplexityB"
out_path = "approx_perplexityB"

sentences_length = []
with open(data_path,'r') as f:
    for line in f:
        tokens = line.split()
        sentences_length.append(len(tokens) + 1)

with open(perp_path, 'r') as f:
    with open(out_path,'w') as out:
        perp = np.loadtxt(f)
        for i, p in enumerate(perp):
            avg_prob = math.exp((-math.log(p, 2)) / sentences_length[i])
            approx_prob = math.pow(2, - sentences_length[i] * math.log(avg_prob, 2))
            out.write(str(approx_prob) + "\n")
