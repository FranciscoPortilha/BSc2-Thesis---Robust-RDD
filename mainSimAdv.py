import numpy as np
from src.simulation import powerSimulations


n, r = 250, 10
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

np.random.seed(123456)
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
