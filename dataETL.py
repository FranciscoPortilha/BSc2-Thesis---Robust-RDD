from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from src.sample import genT

df_83 = pd.read_csv("application/rawData/LinkCO83USden.dat", header=None)
df_84 = pd.read_csv("application/rawData/LinkCO84USden.dat", header=None)
df_85 = pd.read_csv("application/rawData/LinkCO85USden.dat", header=None)
df_86 = pd.read_csv("application/rawData/LinkCO86USden.dat", header=None)
df_87 = pd.read_csv("application/rawData/LinkCO87USden.dat", header=None)
df_88 = pd.read_csv("application/rawData/LinkCO88USden.dat", header=None)
df_89 = pd.read_csv("application/rawData/LinkCO89USden.dat", header=None)
df_90 = pd.read_csv("application/rawData/LinkCO90USden.dat", header=None)
df_91 = pd.read_csv("application/rawData/LinkCO91USden.dat", header=None)
df_95 = pd.read_csv("application/rawData/LinkCO95USden.dat", header=None)
df_96 = pd.read_csv("application/rawData/LinkCO96USden.dat", header=None)
df_97 = pd.read_csv("application/rawData/LinkCO97USden.dat", header=None)
df_98 = pd.read_csv("application/rawData/LinkCO98USden.dat", header=None)
df_99 = pd.read_csv("application/rawData/LinkCO99USden.dat", header=None)
df_00 = pd.read_csv("application/rawData/LinkCO00USden.dat", header=None)
df_01 = pd.read_csv("application/rawData/LinkCO01USden.dat", header=None)
df_02 = pd.read_csv("application/rawData/LinkCO02USDEN.dat", header=None)

Deaths = {}
Weights = {}
year_dummies = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
year = 0

# Extract data
for df in df_83, df_84, df_85, df_86, df_87, df_88:
    for i in range(len(df)):
        # Keep observation and covariates if within the bandwidth
        weight = int(df.iloc[i][0][42:46])
        if (1415 <= weight) & (weight <= 1585):
            death = int(df.iloc[i][0][0])
            if (death == 3) | (death == 1):
                # Store running variable and outcome
                Weights = np.append(Weights, weight)
                if death == 1:
                    Deaths = np.append(Deaths, 1)
                elif death == 3:
                    Deaths = np.append(Deaths, 0)
                # Store year dummy
                for j in range(len(year_dummies)):
                    if j == year:
                        year_dummies[j] = np.append(year_dummies[j], 1)
                    else:
                        year_dummies[j] = np.append(year_dummies[j], 0)
    year = year + 1

for df in df_89, df_90, df_91:
    for i in range(len(df)):
        # Keep observation and covariates if within the bandwidth
        weight = int(df.iloc[i][0][78:82])
        if (1415 <= weight) & (weight <= 1585):
            death = int(df.iloc[i][0][0])
            if (death == 3) | (death == 1):
                # Store running variable and outcome
                Weights = np.append(Weights, weight)
                if death == 1:
                    Deaths = np.append(Deaths, 1)
                elif death == 3:
                    Deaths = np.append(Deaths, 0)
                # Store year dummy
                for j in range(len(year_dummies)):
                    if j == year:
                        year_dummies[j] = np.append(year_dummies[j], 1)
                    else:
                        year_dummies[j] = np.append(year_dummies[j], 0)
    year = year + 1

for df in df_95, df_96, df_97, df_98, df_99, df_00, df_01, df_02:
    for i in range(len(df)):
        # Keep observation and covariates if within the bandwidth
        weight = int(df.iloc[i][0][80:84])
        if (1415 <= weight) & (weight <= 1585):
            Weights = np.append(Weights, weight)
            # Store running variable
            death = int(df.iloc[i][0][0])
            # Store outcome
            if death == 2:
                Deaths = np.append(Deaths, 0)
            else:
                Deaths = np.append(Deaths, 1)
            # Store year dummy
            for j in range(len(year_dummies)):
                if j == year:
                    year_dummies[j] = np.append(year_dummies[j], 1)
                else:
                    year_dummies[j] = np.append(year_dummies[j], 0)
    year = year + 1

# Adjust arrays and create dataframe
Deaths = np.delete(Deaths, 0)
Weights = np.delete(Weights, 0)
for j in range(len(year_dummies)):
    year_dummies[j] = np.delete(year_dummies[j], 0)
sample = pd.DataFrame(
    {
        "Y": Deaths,
        "X": Weights,
        "Treatment": genT(Weights, 1500, False),
        "1983": year_dummies[0],
        "1984": year_dummies[1],
        "1985": year_dummies[2],
        "1986": year_dummies[3],
        "1987": year_dummies[4],
        "1988": year_dummies[5],
        "1989": year_dummies[6],
        "1990": year_dummies[7],
        "1991": year_dummies[8],
        "1995": year_dummies[9],
        "1996": year_dummies[10],
        "1997": year_dummies[11],
        "1998": year_dummies[12],
        "1999": year_dummies[13],
        "2000": year_dummies[14],
        "2001": year_dummies[15],
        "2002": year_dummies[16],
    }
)
sample.Y = sample.Y.astype(float)
sample.X = sample.X.astype(float)
sample.Treatment = sample.Treatment.astype(float)

# Save data
fig = plt.figure()
ax = fig.subplots()
ax.hist(sample.X, bins=85)
fig.savefig("images/application/histApplication.png")
sample.to_csv("application/data.csv", index=False)

# Transform data
Y_avgs = {}
X = {}
for i in range(1415, 1586):
    Y_avgs = np.append(Y_avgs, sample.loc[sample.X == i].mean().iloc[0])
    X = np.append(X, i)

Y_avgs = np.delete(Y_avgs, 0)
X = np.delete(X, 0)
sampleTrans = pd.DataFrame(
    {
        "Y": Y_avgs,
        "X": X,
        "Treatment": genT(X, 1500, False),
    }
)
sampleTrans.Y = sampleTrans.Y.astype(float)
sampleTrans.X = sampleTrans.X.astype(float)
sampleTrans.Treatment = sampleTrans.Treatment.astype(float)

# Save transformes data
fig = plt.figure()
ax = fig.subplots()
ax.scatter(sampleTrans.X, sampleTrans.Y, s=5)
fig.savefig("images/application/ScatterApplication.png")
sampleTrans.to_csv("application/dataTransformed.csv", index=False)
