import src.sample as smp
import src.rrdd as rrdd
import src.simulation as sim
import src.simMetrics as met
import src.exports as exp
import statsmodels.api as sm
import pandas as pd
import numpy as np
import rdrobust

### Test sample
sample = smp.genSample('Basic Linear',1000,tau=-1,alpha=0.5,beta=1,L=0,outlier=False,outlierMethod='Simple Outside', nOutliers=5,printPlot=False)
res = rdrobust.rdrobust(sample.Y, sample.X,0)
res_2 = rrdd.jointFitRD('OLS',sample,b=res.bws.left.iloc[0])
print(res)
print(res.bws)
print(res.bws.left.iloc[0])
print(res.coef)
print(res_2.summary())
print(res_2.params)

# res = rrdd.jointFitRD('OLS',sample)
# print(res.summary())
# print(res.t_test(([0, 0, 1, 0], 0)))
# print(res.t_test(([0, 0, 1, 0], 0), use_t=False))

### Test analytic equality between one and two different regressions
# print(rrdd.splitFitRD('Robust Huber',sample))

# print(sample)
# print(res.scale)

# print(res.pvalues)
# print(res.t_test(([0,0,1,0],0.3)))
# print(res.t_test(([0,0,1,0],0.3)).pvalue)


### Test simulation on one scenario
# p, tv, cic = sim.simulation(10,'Basic Linear',250,tau=2,alpha=-1,beta=1,outlier=False)
# print(met.compRMSE(p,2))
# print('TV')
# print(met.percentV(tv))
# print('CIC')
# print(cic)
#
##
# exp.scenariosHist(p, True, 'images/testfig1.png')
#
# r = 2
# n = 250
#
# simResults = sim.simulations(r, "Basic Linear", n, tau=-1, alpha=0.5, beta=1, cutoff=0)
# met.analyseSimResults(simResults)
