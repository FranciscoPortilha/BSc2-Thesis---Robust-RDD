import methods.sample as smp
import methods.rrdd as rrdd
import methods.simulation as sim
import methods.exports as exp

sample = smp.genSample('Basic Linear',250,tau=0.3,alpha=-0.15,beta=1,L=200,
                   outlier=False,outlierMethod='Simple Outside', nOutliers=5,printPlot=False)

res = rrdd.jointFitRD('Robust Huber',sample)
print(rrdd.splitFitRD('Robust Huber',sample))


print(sample)
print(res.summary())


s = sim.simulation(10,'Basic Linear',30,tau=2,alpha=-1,beta=1,outlier=False)
print(s.OLS)

exp.niceHist(s.OLS).savefig('testfig1')
