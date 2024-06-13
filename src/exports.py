import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.stattools as st
from scipy.stats import norm


def toLatexTable(results1,results2,r,n,):
    '''
    This method prints latex code for a table with the results.

    '''
    # Print results to latex tables
    print("TABLE 1 - ")
    print(results1.to_latex(float_format="{:.3f}".format, multicolumn_format='c'))
    print(results2.to_latex(float_format="{:.3f}".format, multicolumn_format='c'))
    print('r = ' +str(r)+' , n = '+str(n))



def scenariosHist(series, saveFig = False, figPath = ''):
    '''
    This function plot the histogram of the serie with a pdf of a normal function with equal mean and st.dev.

    Parameters
    ----------
    serie: arr[int]
        The serie to plot the histogram of.
    saveFig: boolean
        Determines if the figure is saved or returned
    figPath: string
        The path to print the histogram to.
    JB: boolean, default:False
        Determine if the Jarque-Bera statistics are printed.
    '''
    fig, axs = plt.subplots(2,3,figsize=[20,12])
    labels = ['OLS','Huber','Tukey','Donut']
    colors = ['darkorange','royalblue','forestgreen','purple']
    j ,l = 0,0
    for i in range(6):
        c = 0
        # Plot the histogram 
        for column in series[i]:
            axs[j][l].hist(series[i][column],
                    bins=40,
                    density=True,
                    label=labels[c],
                    zorder=5,
                    edgecolor="k",
                    alpha=0.5,
                    color= colors[c])
            
            # Plot kerndel density function 
            kde = sm.nonparametric.KDEUnivariate(series[i][column])
            kde.fit()
            axs[j][l].plot(kde.support, kde.density, lw=3, zorder=10, color= colors[c])
            c = c+1
        
        axs[j][l].set_title("Scenario " + str(1+i))
        if i==0:
            axs[j][l].legend(loc='upper left')
        if i==2:
            j = 1
            l = -1
        l = l+1

    if saveFig:
        fig.savefig(figPath)
    else:
        return fig
    


