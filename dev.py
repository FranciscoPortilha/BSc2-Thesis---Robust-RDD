import src.sample as smp
import src.rrdd as rrdd
import src.simulation as sim
import src.simMetrics as met
import src.exports as exp
import statsmodels.api as sm
import pandas as pd

### Test sample
#sample = smp.genSample('Basic Linear',250,tau=0.3,alpha=-0.15,beta=1,L=200,outlier=False,outlierMethod='Simple Outside', nOutliers=5,printPlot=False)
#res = rrdd.jointFitRD('Robust Huber',sample)
#print(res.summary())

### Test analytic equality between one and two different regressions
#print(rrdd.splitFitRD('Robust Huber',sample))

#print(sample)
#print(res.scale)

#print(res.conf_int()[1][2])

#print(res.pvalues)
#print(res.t_test(([0,0,1,0],0.3)))
#print(res.t_test(([0,0,1,0],0.3)).pvalue)


### Test simulation on one scenario
#p, tv, cic = sim.simulation(10,'Basic Linear',250,tau=2,alpha=-1,beta=1,outlier=False)
#print(met.compRMSE(p,2))
#print('TV')
#print(met.percentV(tv))
#print('CIC')
#print(met.percentV(cic))
#
##
#exp.scenariosHist(p, True, 'images/testfig1.png')

r = 10
n = 100

results1, results2 = sim.simulations(r,'Basic Linear',n,tau=2,alpha=-1,beta=1,cutoff=0)