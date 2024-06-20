import numpy as np
import pandas as pd
from src.sample import genT
from src.rrdd import jointFitRD
import rdrobust

df_83 = pd.read_csv("application/LinkCO83USden.dat", header=None)
#df_84 = pd.read_csv("application/LinkCO84USden.dat", header=None)
#df_85 = pd.read_csv("application/LinkCO85USden.dat", header=None)
#df_86 = pd.read_csv("application/LinkCO86USden.dat", header=None)
#df_87 = pd.read_csv("application/LinkCO87USden.dat", header=None)
#df_88 = pd.read_csv("application/LinkCO88USden.dat", header=None)
#df_89 = pd.read_csv("application/LinkCO89USden.dat", header=None)
#df_90 = pd.read_csv("application/LinkCO90USden.dat", header=None)
#df_90 = pd.read_csv("application/LinkCO90USden.dat", header=None)
#df_91 = pd.read_csv("application/LinkCO91USden.dat", header=None)
#df_95 = pd.read_csv("application/LinkCO95USden.dat", header=None)
#df_96 = pd.read_csv("application/LinkCO96USden.dat", header=None)
#df_97 = pd.read_csv("application/LinkCO97USden.dat", header=None)
#df_98 = pd.read_csv("application/LinkCO98USden.dat", header=None)
#df_99 = pd.read_csv("application/LinkCO99USden.dat", header=None)
#df_00 = pd.read_csv("application/LinkCO00USden.dat", header=None)
#df_01 = pd.read_csv("application/LinkCO01USden.dat", header=None)
#df_02 = pd.read_csv("application/LinkCO02USDEN.dat", header=None)

Deaths = {}
Weights = {}
print(df_83.iloc[1][0][0:50])
print(df_83.iloc[1][0][42:46])
#df_85, df_86, df_87, df_88, df_89, df_90, df_91, df_02:
#for df in df_83:
for i in range(len(df_83)):
    #weight = int(df_83.iloc[i][0][80:84])
    weight = int(df_83.iloc[i][0][42:46])
    death = int(df_83.iloc[i][0][0])
    if (1415 <= weight) & (weight <= 1585):
        Weights = np.append(Weights, weight)
        if(death==3):
            Deaths = np.append(Deaths, 0)
        else:
            Deaths = np.append(Deaths, 1)
        

Deaths = np.delete(Deaths, 0)
Weights = np.delete(Weights, 0)

print(Deaths)
print(Weights)
print(genT(Weights, 1500))

sample = pd.DataFrame(
    {
        "Y": Deaths,
        "X": Weights,
        "Treatment": genT(Weights),
        "Outlier": np.zeros_like(Deaths),
    }
)
sample.Y = sample.Y.astype(float)
sample.X = sample.X.astype(float)
sample.Treatment = sample.Treatment.astype(float)
sample.Outlier = sample.Outlier.astype(float)

print(jointFitRD("OLS",sample,cutoff=1500,b=85).summary())

rdrobust.rdplot(sample.Y, sample.X, 1500, 1, 60)