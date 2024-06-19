import numpy as np
from src.simulation import simulations

n, r = 1000, 100

np.random.seed(234567)
L = [0, 10, 20, 30, 40,200]

simulations(r, "Noack", n, L=L, cutoff=0, b=0.5, printToLatex=False)
