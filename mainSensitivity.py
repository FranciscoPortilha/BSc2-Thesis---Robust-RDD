import numpy as np
from src.simulation import simulations

n, r = 250, 100
outlierScenarios = (
    "Outside Right",
    2,
    "Outside Right",
    4,
    "Outside Right",
    8,
    "Outside Right",
    16,
    "Outside Right",
    32,
)

np.random.seed(3456)
for t in [-0.5]:
    simulations(
        r,
        "Basic Linear",
        n,
        tau=t,
        alpha=0,
        beta=1,
        cutoff=0,
        parametersScenarios=outlierScenarios,
        printToLatex=True,
        figureFolder="sensitivity",
    )
