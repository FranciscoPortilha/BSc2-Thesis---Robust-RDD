import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.stats.stattools as st
from scipy.stats import norm


def latexTable(results1,results2,r,n,):
    '''
    This method prints latex code for a table with the results
    
    '''
    # Print results to latex tables
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
    hist = plt.hist(serie, bins=60, density=True)
    plt.title("Fit Values: {:.2f} and {:.2f}".format(np.mean(serie), np.std(serie)))

    # Plot the matching normal PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100) 
    p = norm.pdf(x, np.mean(serie), np.std(serie))
    plt.plot(x, p, 'k', linewidth=2)