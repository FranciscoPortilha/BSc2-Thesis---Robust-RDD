import numpy as np
import pandas as pd
from src.sample import genT
from src.rrdd import jointFitRD
from src.exports import plotApplicationFigure
import rdrobust

sample = pd.read_csv("application/dataTransformed.csv")
sample = sample[["Y","X","Treatment"]]
print(sample)
plotApplicationFigure(sample,cutoff=1500)




print(jointFitRD("OLS", sample, cutoff=00, b=85, outliers=False).summary())
print(jointFitRD("Donut", sample, cutoff=0, b=85, outliers=False,donut=3.1).summary())
print(jointFitRD("Robust Huber", sample, cutoff=00, b=85, outliers=False).summary())
print(jointFitRD("Robust Tukey", sample, cutoff=00, b=85, outliers=False).summary())