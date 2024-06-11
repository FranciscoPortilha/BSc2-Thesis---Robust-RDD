import methods.exports as exp
from methods.simulation import simulations


r = 10
n = 300

results1, results2 = simulations(r,'Basic Linear',n,tau=2,alpha=-1,beta=1,cutoff=0)
exp.toLatexTable(results1, results2,r,n)
