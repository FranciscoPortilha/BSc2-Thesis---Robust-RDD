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



def niceHist(series, saveFig = False, figPath = '', JB = False):
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
    # print jarque-bera statistics
    if JB == True:
        print('Jarque-Bera : '+ st.jarque_bera(serie))

    # Plot the histogram    
    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(1,1,1)
    labels = ['OLS','Huber','Tukey','Donut']
    colors = ['r','b','g','purple']
    c = 0
    for column in series:
        print(series[column])
        ax.hist(series[column],
                bins=40,
                density=True,
                label=labels[c] +" - estimates",
                zorder=5,
                edgecolor="k",
                alpha=0.5,
                color= colors[c])
        #ax.title("Fit Values: {:.2f} and {:.2f}".format(np.mean(serie), np.std(serie)))
        
        # Plot kerndel density function 
        kde = sm.nonparametric.KDEUnivariate(series[column])
        kde.fit()
        ax.plot(kde.support, kde.density, lw=3, label=labels[c] + ' - kde', zorder=10, color= colors[c])
        c = c+1
    # Export histogram to path or return
    ax.legend()
    if saveFig:
        fig.savefig(figPath)
    else:
        return fig