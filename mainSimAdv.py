import numpy as np
from src.simulation import powerSimulations


n, r = 250, 1000
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

np.random.seed(2345)
powerSimulations(
    r,
    "Basic Linear",
    n,
    alpha=0.5,
    beta=1,
    cutoff=0,
    parametersScenarios=outlierScenarios,
    specialTau=[-0.5, 0],
    computeAsymptotics=True,
    prinToLatex=False,
)
