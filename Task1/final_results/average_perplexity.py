import numpy as np

path = "group02.perplexityA"

with open(path) as f:
    perp = np.loadtxt(f)
    print(np.mean(perp))