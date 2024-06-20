import numpy as np
from src.simulation import powerSimulations


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
