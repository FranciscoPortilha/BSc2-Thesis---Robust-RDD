import numpy as np
from src.simulation import simulations

n, r = 250, 1000
outlierScenarios = (
    "Small Outside Right",
    2,
    "Outside Right",
    3,
    "Oposite Outside",
    3,
    "Oposite Inside",
    3,
    "Symetric Inside",
    2,
)

np.random.seed(234567)
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
        printToLatex=False,
        figureFolder="base",
    )
