import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.stattools as st


def toLatexTable(results1, results2="", caption="", ref=""):
    """
    This method prints latex code for a table with the results.

    """
    # Print results to latex tables
    print(
        results1.to_latex(
            float_format="{:.3f}".format,
            multicolumn_format="c",
            caption=caption,
            label=ref,
        )
    )
    if any(results2):
        print(
            results2.to_latex(
                float_format="{:.3f}".format,
                multicolumn_format="c",
                caption=caption,
                label=ref,
            )
        )


def plotScenariosHist(scenarios, saveFig=False, figPath=""):
    """
    This function plots the histograms of the estimated ATE from the different 
    scenarios, with a kernel density estimation.    

    Parameters
    ----------
    scenarios: arr[int]
        The scenarios to plot the histograms of.
    saveFig: boolean
        Determines if the figure is saved or returned
    figPath: string
        The path to print the histogram to.
    """
    fig, axs = plt.subplots(2, 3, figsize=[20, 12])
    labels = ["OLS", "Huber", "Tukey", "Donut"]
    colors = ["darkorange", "royalblue", "mediumseagreen", "mediumorchid"]
    j, l = 0, 0
    # For each scenario
    for i in range(6):
        c = 0
        # Plot the histogram
        for model in scenarios[i]:
            
            axs[j][l].hist(
                scenarios[i][model],
                bins=40,
                density=True,
                label=labels[c],
                zorder=5,
                edgecolor="k",
                alpha=0.5,
                color=colors[c],
            )
            axs[j][l].set_xlabel("$\^τ$",fontsize=10)
            axs[j][l].set_ylabel("frequency")
            # Plot kerndel density function
            kde = sm.nonparametric.KDEUnivariate(scenarios[i][model])
            kde.fit()
            axs[j][l].plot(kde.support, kde.density, lw=3, zorder=10, color=colors[c])
            c = c + 1

            

        axs[j][l].set_title("Scenario " + str(1 + i))
        if i == 0:
            axs[j][l].legend(loc="upper left")
        if i == 2:
            j = 1
            l = -1
        l = l + 1

    if saveFig:
        fig.savefig(figPath)
    else:
        return fig


def plotSamplesComparison(samples, saveFig=False, figPath="",cutoff=0):
    """
    This method plots a figure with the regession lines of the different estimation methods,
    for all scenarios.

    Parameters
    ----------
    sample : DataFrame
        The sample to estimate the regression for.
    saveFig: boolean
        Determines if the figure is saved or returned
    figPath: string
        The path to print the histogram to.
    cutoff : int
        The value of the threshold in the running variable.
    """
    fig, axs = plt.subplots(2, 3, figsize=[20, 12])
    labels = ["OLS", "Huber", "Tukey", "Donut"]
    colors = ["darkorange", "royalblue", "mediumseagreen", "mediumorchid"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["grey", "red"])
    j, l = 0, 0
    # For each scenario
    for i in range(6):
        # Plot scatter observations
        axs[j][l].scatter(samples[i].X, samples[i].Y, s=6, c=samples[i].Outlier, cmap=cmap)
        axs[j][l].set_xlabel("$X_i$")
        axs[j][l].set_ylabel("$Y_i$")

        
        #x_below = np.linspace(min(samples[i].X), cutoff, 100)
        #x_above = np.linspace(cutoff, max(samples[i].X), 100)
        ## Plot regresion lines
        #for model in samples[i][1]:
        #    c=0
        #    plt.plot(
        #        x_below,
        #        params1_below.iloc[0] + params1_below.iloc[1] * x_below,
        #        color=colors[c],
        #        linewidth=0.7,
        #    )
        #    plt.plot(
        #        x_above,
        #        params1_above.iloc[0] + params1_above.iloc[1] * x_above,
        #        color="b",
        #        linewidth=0.7,
        #        label=labels[c] + " (ate: " + str(round(tau1, 2)) + ")",
        #    )
        #
        #
        #
        #plt.legend()
        
        axs[j][l].set_title("Scenario " + str(1 + i))
        if i == 0:
            axs[j][l].legend(loc="upper left")
        if i == 2:
            j = 1
            l = -1
        l = l + 1

    if saveFig:
        fig.savefig(figPath)
    else:
        return fig

