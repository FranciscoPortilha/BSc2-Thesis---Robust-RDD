import numpy as np
import pandas as pd
from src.sample import genT
from src.rrdd import jointFitRD
import rdrobust

sample = pd.read_csv("application/data.csv")

Y_avgs = {}
X = {}
for i in range(1415,1586):
    Y_avgs = np.append(Y_avgs, sample.loc[sample.X==i].mean().iloc[1])
    X = np.append(X, i)

Y_avgs = np.delete(Y_avgs,0)
X = np.delete(X,0)
sample = pd.DataFrame(
    {
        "Y": Y_avgs,
        "X": X,
        "Treatment": genT(X,1500,False),
    }
)
sample.Y = sample.Y.astype(float)
sample.X = sample.X.astype(float)
sample.Treatment = sample.Treatment.astype(float)


print(jointFitRD("OLS", sample, cutoff=1500, b=85, outliers=False).summary())
print(jointFitRD("Donut", sample, cutoff=1500, b=85, outliers=False,donut=3).summary())
print(jointFitRD("Robust Huber", sample, cutoff=1500, b=85, outliers=False).summary())
print(jointFitRD("Robust Tukey", sample, cutoff=1500, b=85, outliers=False).summary())

rdrobust.rdplot(sample.Y, sample.X, 1500, 1, 60).savefig("images/application/rdplot")
