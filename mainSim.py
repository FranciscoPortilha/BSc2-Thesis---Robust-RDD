import numpy as np
from src.simulation import simulations

n, r = 250, 10
outlierScenarios = (
    "Simple Outside Right",
    1,
    "Simple Oposite",
    3,
    "Simple Outside Left",
    2,
    "Simple Outside Right",
    3,
    "Simple Oposite Inside",
    3,
)

np.random.seed(234567)
for t in -0.5, 0:
    simulations(
        r,
        "Basic Linear",
        n,
        tau=t,
        alpha=0.5,
        beta=1,
        cutoff=0,
        parametersScenarios=outlierScenarios,
        printToLatex=False,
    )
