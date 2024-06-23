"""
Code for my bachelor thesis in Econometrics and Economics, 
Outlier Robust Regression Discontinuity Designs.

Author: Francisco Portilha (479126)
"""

# Public libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# My methods. Developed for my thesis.
from src.rrdd import jointFitRD


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


def plotScenariosHist(scenarios, tau, saveFig=False, figPath=""):
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
            axs[j][l].set_xlabel(r"$\^τ$", fontsize=10)
            axs[j][l].set_ylabel("frequency")
            # Plot kerndel density function
            kde = sm.nonparametric.KDEUnivariate(scenarios[i][model])
            kde.fit()
            axs[j][l].plot(kde.support, kde.density, lw=3, zorder=10, color=colors[c])
            c = c + 1
        axs[j][l].axvline(x=tau, color="r")
        axs[j][l].set_title("Scenario " + str(1 + i))

        # Increment figure location and add legend
        if i == 0:
            axs[j][l].legend(loc="upper left")
        if i == 2:
            j = 1
            l = -1
        l = l + 1

    # Save figure
    if saveFig:
        fig.savefig(figPath)
        plt.close()
    else:
        return fig


def plotSamplesComparison(
    samples, saveFig=False, figPath="", printRegLines=False, cutoff=0, b=1
):
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
    models = ["OLS", "Robust Huber", "Robust Tukey", "Donut"]
    labels = ["OLS", "Huber", "Tukey", "Donut"]
    colors = ["darkorange", "royalblue", "mediumseagreen", "mediumorchid"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["grey", "red"])
    j, l = 0, 0
    # For each scenario
    for i in range(6):
        # Plot scatter observations
        axs[j][l].scatter(
            samples[i].X, samples[i].Y, s=4, c=samples[i].Outlier, cmap=cmap
        )
        axs[j][l].set_xlabel("$X_i$")
        axs[j][l].set_ylabel("$Y_i$")

        # (Re-fit) and plot regression lines
        if printRegLines:
            x_below = np.linspace(min(samples[i].X), cutoff, 100)
            x_above = np.linspace(cutoff, max(samples[i].X), 100)
            c = 0
            # For each estimation method
            for model in models:
                # Fit model
                res = jointFitRD(model, samples[i], cutoff, b)
                # print(res.summary())
                params = res.params

                # Plot regression below cutoff
                axs[j][l].plot(
                    x_below,
                    params.iloc[0] + params.iloc[1] * x_below,
                    color=colors[c],
                    linewidth=0.7,
                )
                # Plot regression above cutoff
                axs[j][l].plot(
                    x_above,
                    (params.iloc[0] + params.iloc[2])
                    + (params.iloc[1] + params.iloc[3]) * x_above,
                    color=colors[c],
                    linewidth=0.7,
                    label=labels[c]
                    + r" ($\^τ$: "
                    + str(round(params.iloc[2], 2))
                    + "("
                    + "{:.2f}".format(round(res.pvalues.iloc[2], 2))
                    + "))",
                )
                # Increment model label counter
                c = c + 1

            # Legend and title
            axs[j][l].legend(loc="upper left")
        axs[j][l].set_title("Scenario " + str(1 + i))

        # Increment figure location
        if i == 2:
            j = 1
            l = -1
        l = l + 1

    # Save figure
    if saveFig:
        fig.savefig(figPath)
        plt.close()

    else:
        return fig


def plotPowerFunctionComparison(taus, rejectionRates, saveFig=False, figPath=""):
    # Initialise figure, labels and colors
    fig, axs = plt.subplots(2, 3, figsize=[19, 11])
    labels = ["OLS", "Huber", "Tukey", "Donut"]
    colors = ["darkorange", "royalblue", "mediumseagreen", "mediumorchid"]

    # Vertical and horizontal subplot location
    v, h = 0, 0

    # For each scenario
    for scenario in range(6):
        # Plot the power functions
        for model in range(len(labels)):
            axs[v][h].plot(
                taus,
                rejectionRates[scenario][model],
                color=colors[model],
                label=labels[model],
                linewidth=0.8,
            )
        axs[v][h].plot(taus, 0.05 + np.zeros_like(taus), color="r", linewidth=0.6)
        axs[v][h].set_xlim([-2.25, 2.25])
        axs[v][h].set_ylim([-0.05, 1.05])
        # Add axis labels and title
        axs[v][h].set_ylabel("rejection rate")
        axs[v][h].set_xlabel("$τ$")
        axs[v][h].set_title("Scenario " + str(1 + scenario))

        # Increment figure location and add lengend
        if scenario == 0:
            axs[v][h].legend(loc="upper left")
        if scenario == 2:
            v = 1
            h = -1
        h = h + 1

    # Save figure
    if saveFig:
        fig.savefig(figPath)
        plt.close()
    else:
        return fig


def plotAsymptoticComparison(tau, asymptotics, saveFig=False, figPath=""):
    fileLabels = (
        "bias",
        "stDev",
        "rmse",
        "efficiency",
        "ciCc",
        "ciSize",
        "t1Error",
        "t2Error",
    )
    plotyLabels = (
        r"Absoulute bias($\^τ$)",
        r"St.dev($\^τ$)",
        r"RMSE($\^τ$)",
        "Efficiency relative to Tukey",
        "Correct coverage of C.I.",
        "Length of C.I.",
        "T.I rejection rate",
        "T.II acceptance rate",
    )

    # For each metric plot asymptotic functions
    for metric in range(8):
        # Initialise figure, labels and colors
        fig, axs = plt.subplots(2, 3, figsize=[19, 11])
        labels = ["OLS", "Huber", "Tukey", "Donut"]
        colors = ["darkorange", "royalblue", "mediumseagreen", "mediumorchid"]

        # Vertical and horizontal subplot location
        v, h = 0, 0

        # For each scenario
        for scenario in range(6):
            # Plot the asymptotic function
            for model in range(len(labels)):
                if (metric == 3) & (model == 2):
                    0
                elif metric == 0:
                    axs[v][h].plot(
                        asymptotics[0][8],
                        np.abs(asymptotics[scenario][metric][model]),
                        color=colors[model],
                        label=labels[model],
                        linewidth=0.8,
                    )
                else:
                    axs[v][h].plot(
                        asymptotics[0][8],
                        asymptotics[scenario][metric][model],
                        color=colors[model],
                        label=labels[model],
                        linewidth=0.8,
                    )
            # Plot critical line at 0 for bias and rmse
            if (metric == 0) | (metric == 2):
                axs[v][h].plot(
                    asymptotics[0][8],
                    np.zeros_like(asymptotics[0][8]),
                    color="r",
                    linewidth=0.6,
                )
            # Plot critical line at 0.05 for type 1 nd 2 errors
            elif (metric == 6) | (metric == 7):
                axs[v][h].plot(
                    asymptotics[0][8],
                    0.05 + np.zeros_like(asymptotics[0][8]),
                    color="r",
                    linewidth=0.6,
                )
            # Plot critical line at 0.95 for correct coverage
            elif metric == 4:
                axs[v][h].plot(
                    asymptotics[0][8],
                    0.95 + np.zeros_like(asymptotics[0][8]),
                    color="r",
                    linewidth=0.6,
                )
            # Plot critical line at 1 for relative efficiency
            elif metric == 3:
                axs[v][h].plot(
                    asymptotics[0][8],
                    1 + np.zeros_like(asymptotics[0][8]),
                    color="r",
                    linewidth=0.6,
                )

            # Set axis scale to log and add labels and title
            axs[v][h].set_xscale("log")
            if metric in [0, 1, 2, 5]:
                axs[v][h].set_yscale("log")
            axs[v][h].set_ylabel(plotyLabels[metric])
            axs[v][h].set_xlabel("Number of observations")
            axs[v][h].set_title("Scenario " + str(1 + scenario))

            # Increment figure location and add lengend
            if scenario == 0:
                axs[v][h].legend(loc="upper left")
            if scenario == 2:
                v = 1
                h = -1
            h = h + 1

        # Save figure
        if saveFig:
            fig.savefig(figPath + "_" + fileLabels[metric] + "_" + str(tau) + ".png")
            plt.close()
        else:
            return fig


def plotApplicationFigure(sample, cutoff=0, b=85, d=3.1):
    models = ["OLS", "Robust Huber", "Robust Tukey", "Donut"]
    labels = ["OLS", "Huber", "Tukey", "Donut"]
    colors = ["darkorange", "royalblue", "mediumseagreen", "mediumorchid"]
    fig = plt.figure()
    ax = fig.subplots()
    ax.scatter(sample.X, sample.Y, s=5)
    ax.plot()
    x_below = np.linspace(min(sample.X), cutoff, 100)
    x_above = np.linspace(cutoff, max(sample.X), 100)
    sample.X = sample.X - cutoff
    c = 0
    for model in models:
        # sample.X = sample.X-cutoff
        res = jointFitRD(model, sample, cutoff=0, b=85, outliers=False, donut=d)
        params = res.params
        ax.plot(
            x_below,
            (params.iloc[0] + params.iloc[2])
            + (params.iloc[1] + params.iloc[3]) * (x_below - cutoff),
            color=colors[c],
            linewidth=0.7,
        )
        ax.plot(
            x_above,
            params.iloc[0] + params.iloc[1] * (x_above - cutoff),
            color=colors[c],
            linewidth=0.7,
            label=labels[c]
            + r" ($\^τ$: "
            + "{:.3f}".format(round(params.iloc[2], 3))
            + "("
            + "{:.2f}".format(round(res.pvalues.iloc[2], 2))
            + "))",
        )
        c = c + 1
    ax.legend()

    fig.savefig("images/application/figureApplication.png")
