import numpy as np
from src.simulation import simulations

n, r = 250, 10000
outlierScenarios = (
    "Small Outside Right",
    2,
    "Outside Right",
    2,
    "Oposite Outside",
    2,
    "Oposite Inside",
    2,
    "Symetric Inside",
    1,
)

np.random.seed(1234)
for t in [-0.5,0]:
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
        figureFolder="base",
    )
