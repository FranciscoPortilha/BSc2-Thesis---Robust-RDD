from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from src.sample import genT

#df_83 = pd.read_csv("application/rawData/LinkCO83USden.dat", header=None)
#df_84 = pd.read_csv("application/rawData/LinkCO84USden.dat", header=None)
#df_85 = pd.read_csv("application/rawData/LinkCO85USden.dat", header=None)
#df_86 = pd.read_csv("application/rawData/LinkCO86USden.dat", header=None)
#df_87 = pd.read_csv("application/rawData/LinkCO87USden.dat", header=None)
#df_88 = pd.read_csv("application/rawData/LinkCO88USden.dat", header=None)
#df_89 = pd.read_csv("application/rawData/LinkCO89USden.dat", header=None)
#df_90 = pd.read_csv("application/rawData/LinkCO90USden.dat", header=None)
#df_91 = pd.read_csv("application/rawData/LinkCO91USden.dat", header=None)
#df_95 = pd.read_csv("application/rawData/LinkCO95USden.dat", header=None)
#df_96 = pd.read_csv("application/rawData/LinkCO96USden.dat", header=None)
#df_97 = pd.read_csv("application/rawData/LinkCO97USden.dat", header=None)
#df_98 = pd.read_csv("application/rawData/LinkCO98USden.dat", header=None)
#df_99 = pd.read_csv("application/rawData/LinkCO99USden.dat", header=None)
#df_00 = pd.read_csv("application/rawData/LinkCO00USden.dat", header=None)
#df_01 = pd.read_csv("application/rawData/LinkCO01USden.dat", header=None)
df_02 = pd.read_csv("application/rawData/LinkCO02USDEN.dat", header=None)

Deaths = {}
Weights = {}

## Extract data
#for df in df_83,df_84,df_85, df_86, df_87, df_88:
#    for i in range(len(df)):
#        weight = int(df.iloc[i][0][42:46])
#        death = int(df.iloc[i][0][0])
#        if (1415 <= weight) & (weight <= 1585):
#            Weights = np.append(Weights, weight)
#            if death == 3:
#                Deaths = np.append(Deaths, 0)
#            else:
#                Deaths = np.append(Deaths, 1)
#
#for df in df_89, df_90, df_91:
#    for i in range(len(df)):
#        weight = int(df.iloc[i][0][78:82])
#        death = int(df.iloc[i][0][0])
#        if (1415 <= weight) & (weight <= 1585):
#            Weights = np.append(Weights, weight)
#            if death == 3:
#                Deaths = np.append(Deaths, 0)
#            else:
#                Deaths = np.append(Deaths, 1)

for df in [df_02]: # df_96, df_97, df_98, df_99, df_00, df_01, df_02:
    for i in range(len(df)):
        weight = int(df.iloc[i][0][80:84])
        death = int(df.iloc[i][0][0])
        if (1415 <= weight) & (weight <= 1585):
            Weights = np.append(Weights, weight)
            if death == 2:
                Deaths = np.append(Deaths, 0)
            else:
                Deaths = np.append(Deaths, 1)


Deaths = np.delete(Deaths, 0)
Weights = np.delete(Weights, 0)

sample = pd.DataFrame(
    {
        "Y": Deaths,
        "X": Weights,
        "Treatment": genT(Weights,1500,False),
    }
)
sample.Y = sample.Y.astype(float)
sample.X = sample.X.astype(float)
sample.Treatment = sample.Treatment.astype(float)

# Save data
fig = plt. figure()
fig.hist(sample.X, bins= 85)
fig.savefig("images/application/histApplication.png")
sample.to_csv("application/data.csv")

# Transform data
Y_avgs = {}
X = {}
for i in range(1415,1586):
    Y_avgs = np.append(Y_avgs, sample.loc[sample.X==i].mean().iloc[0])
    X = np.append(X, i)

Y_avgs = np.delete(Y_avgs,0)
X = np.delete(X,0)
sampleTrans = pd.DataFrame(
    {
        "Y": Y_avgs,
        "X": X,
        "Treatment": genT(X,1500,False),
    }
)
sampleTrans.Y = sampleTrans.Y.astype(float)
sampleTrans.X = sampleTrans.X.astype(float)
sampleTrans.Treatment = sampleTrans.Treatment.astype(float)

# Save data
plt.scatter(sampleTrans.X, sampleTrans.Y, s=5)
sample.to_csv("application/dataTransformed.csv")
