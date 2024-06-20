import numpy as np
from src.simulation import simulations

n, r = 1000, 100

np.random.seed(345678)
L = [0, 10, 20, 30, 40, 200]

simulations(
    r, "Noack", n, L=L, cutoff=0, b=0.35, printToLatex=False, figureFolder="replication"
)
