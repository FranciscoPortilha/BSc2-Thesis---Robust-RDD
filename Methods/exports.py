import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.stattools as st
from scipy.stats import norm


def toLatexTable(results1,results2,r,n,):
    '''
    This method prints latex code for a table with the results

    '''
    # Print results to latex tables
    print("TABLE 1 - ")
    print(results1.to_latex(float_format="{:.3f}".format, multicolumn_format='c'))
    print(results2.to_latex(float_format="{:.3f}".format, multicolumn_format='c'))
    print('r = ' +str(r)+' , n = '+str(n))



def niceHist(serie, JB = False):
    '''
    This function plot the histogram of the serie with a pdf of a normal function with equal mean and st.dev

    Parameters
    ----------
    serie: arr[int]
        the serie to plot the histogram of
    JB: boolean, default:False
        Determine if the Jarque-Bera statistics are printed
    '''
    # print jarque-bera statistics
    if JB == True:
        print('Jarque-Bera : '+ st.jarque_bera(serie))

    # Plot the histogram    
    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(1,1,1)
    ax.hist(serie,
            bins=20,
            density=True,
            label="Histogram from samples",
            zorder=5,
            edgecolor="k",
            alpha=0.5,)
    #ax.title("Fit Values: {:.2f} and {:.2f}".format(np.mean(serie), np.std(serie)))
    
    kde = sm.nonparametric.KDEUnivariate(serie)
    kde.fit()
    
    ax.plot(kde.support, kde.density, lw=3, label="KDE from samples", zorder=10)

    return fig
    # Plot the matching normal PDF.
    #xmin, xmax = plt.xlim()
    #x = np.linspace(xmin, xmax, 100) 
    #p = norm.pdf(x, np.mean(serie), np.std(serie))
    #plt.plot(x, p, 'k', linewidth=2)