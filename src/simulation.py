"""
Code for my bachelor thesis in Econometrics and Economics, 
Outlier Robust Regression Discontinuity Designs.

Author: Francisco Portilha (479126)
"""

# Public libraries
import numpy as np
import pandas as pd

# My methods. Developed for my thesis.
from src.exports import plotSamplesComparison, plotScenariosHist, toLatexTable
from src.rrdd import jointFitRD
from src.sample import genSample
from src.simMetrics import compJB, compKurt, compRMSE, compSkew


def simulation(
    r,
    nameSample,
    n,
    tau=0,
    alpha=0,
    beta=0,
    L=0,
    cutoff=0,
    b=1,
    outlier=False,
    outlierMethod="",
    nOutliers=0,
):
    """
    Run simulation analysis for RDD: Generates a sample r times and returns the results
    of each sample. Also returns the first sample.

    Parameters
    ----------
    r: int
        The number of replications.
    nameSample: string, Options: 'Noak', 'Basic Linear'
        The name of the DGP to use to generate the sample.
    n: int
        The size of the sample.
    tau : int , Default value: 0
        The size of the treatment effect. For basic and basic linear model.
    L : int, Default value: 0
        The parameter used by NR to define the level of misspecification L={0,10,20,30,40}.
        For the Noack and Rothe model.
    cutoff : int , Default value: 0
        The treshold of the running variable that determines treatment
    b : int , Default value: 1
        Represent the bandwidth selected
    alpha: int, Default value: 0
        The intercept parameter of the equation. For basic linear model.
    beta: int, Default value: 0
        The slope parameter of the equation. For basic linear model.

    Returns
    -------
    pointEstimation: DataFrame
        A dataframe object with the point estimation results from the simulation.
        Values are stored as bias relative to the true value.
    testValues: DataFrame
        A dataframe object with the results about the type I error of t-tests.
        Includes two dataframes one with results based on normal and one based on t distr.
        Values are stored as 1 for rejection of a correct null.
    ciValues: DataFrame
        A dataframe object with the results about the confidence intervals.
        Values are stored as 1 for correct coverage.
    firstSample: DataFrame
        A dataframe object with the first sample generated.
    """

    # Creat empty output arrays
    bias = [{}, {}, {}, {}]

    t_normal = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]
    t_studT = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]

    ci_cove = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]
    ci_lenght = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]

    models = ["OLS", "Robust Huber", "Robust Tuckey", "Donut"]
    firstSample = []

    # Generate sample, fit models and record results r times
    for k in range(r):
        # Generate a new sample
        sample = genSample(
            nameSample,
            n,
            tau,
            alpha,
            beta,
            L,
            cutoff,
            outlier,
            outlierMethod,
            nOutliers,
            False,
        )

        # Select part of the sample to be used according to bandwidth
        sample = sample.loc[np.abs(sample.X - cutoff) <= b]

        # Store first sample for comparison
        if k == 0:
            firstSample = sample
        # Estimate the different models and record results
        for i in range(len(models)):
            # Fit the model
            res = jointFitRD(models[i], sample, cutoff)

            # Record point estimate
            bias[i] = np.append(bias[i], res.params.iloc[2] - tau)

            # Record if t test of Ho: t == tau is rejected based on normal and student-t
            useStudentT = False
            for t in t_normal, t_studT:
                if (tau == 0) & (useStudentT == False):
                    p = res.pvalues.iloc[2]
                else:
                    p = res.t_test(([0, 0, 1, 0], tau), use_t=useStudentT).pvalue

                # Record if t test of Ho: t == tau is rejected
                if p >= 0.1:
                    # No incorrect rejections
                    t[0][i] = np.append(t[0][i], 0)
                    t[1][i] = np.append(t[1][i], 0)
                    t[2][i] = np.append(t[2][i], 0)

                elif p >= 0.05:
                    # Incorrect rejection at 0.1 level
                    t[0][i] = np.append(t[0][i], 1)
                    t[1][i] = np.append(t[1][i], 0)
                    t[2][i] = np.append(t[2][i], 0)

                elif p >= 0.01:
                    # Incorrect rejection at 0.1 and 0.05 levels
                    t[0][i] = np.append(t[0][i], 1)
                    t[1][i] = np.append(t[1][i], 1)
                    t[2][i] = np.append(t[2][i], 0)

                else:
                    # Incorrect rejection at all levels
                    t[0][i] = np.append(t[0][i], 1)
                    t[1][i] = np.append(t[1][i], 1)
                    t[2][i] = np.append(t[2][i], 1)

                useStudentT = True

            # Record correct coverage of confidence interval
            if (res.conf_int(0.1)[0].iloc[2] < tau) & (
                tau < res.conf_int(0.1)[1].iloc[2]
            ):
                # Correct coverage all levels
                ci_cove[0][i] = np.append(ci_cove[0][i], 1)
                ci_cove[1][i] = np.append(ci_cove[1][i], 1)
                ci_cove[2][i] = np.append(ci_cove[2][i], 1)

            elif (res.conf_int()[0].iloc[2] < tau) & (tau < res.conf_int()[1].iloc[2]):
                # Incorrect coverage 0.1 level
                ci_cove[0][i] = np.append(ci_cove[0][i], 0)
                ci_cove[1][i] = np.append(ci_cove[1][i], 1)
                ci_cove[2][i] = np.append(ci_cove[2][i], 1)

            elif (res.conf_int(0.01)[0].iloc[2] < tau) & (
                tau < res.conf_int(0.01)[1].iloc[2]
            ):
                # Incorrect coverage 0.1 and 0.5 level
                ci_cove[0][i] = np.append(ci_cove[0][i], 0)
                ci_cove[1][i] = np.append(ci_cove[1][i], 0)
                ci_cove[2][i] = np.append(ci_cove[2][i], 1)

            else:
                # Incorrect coverage on all levels
                ci_cove[0][i] = np.append(ci_cove[0][i], 0)
                ci_cove[1][i] = np.append(ci_cove[1][i], 0)
                ci_cove[2][i] = np.append(ci_cove[2][i], 0)

            # Record lenght of C.I. for all significance levels
            ci_lenght[0][i] = np.append(
                ci_lenght[0][i],
                res.conf_int(0.1)[1].iloc[2] - res.conf_int(0.1)[0].iloc[2],
            )
            ci_lenght[1][i] = np.append(
                ci_lenght[1][i], res.conf_int()[1].iloc[2] - res.conf_int()[0].iloc[2]
            )
            ci_lenght[2][i] = np.append(
                ci_lenght[2][i],
                res.conf_int(0.01)[1].iloc[2] - res.conf_int(0.01)[0].iloc[2],
            )

    # Adjust the format of the arrays (delete empty first cell)
    for i in range(len(models)):
        bias[i] = np.delete(bias[i], 0)
        for t in t_normal, t_studT:
            t[0][i] = np.delete(t[0][i], 0)
            t[1][i] = np.delete(t[1][i], 0)
            t[2][i] = np.delete(t[2][i], 0)
        for c in ci_cove, ci_lenght:
            c[0][i] = np.delete(c[0][i], 0)
            c[1][i] = np.delete(c[1][i], 0)
            c[2][i] = np.delete(c[2][i], 0)

    # Create dataframe with point estimation results from the simulation
    pointEstimation = pd.DataFrame(
        {"OLS": bias[0], "Huber": bias[1], "Tukey": bias[2], "Donut": bias[3]}
    )

    # Create dataframes with t-test results from the simulation
    testValues = ["", ""]
    k = 0
    for t in t_normal, t_studT:
        testValues_1 = pd.DataFrame(
            {
                "OLS": t[0][0],
                "Huber": t[0][1],
                "Tukey": t[0][2],
                "Donut": t[0][3],
            }
        )
        testValues_05 = pd.DataFrame(
            {
                "OLS": t[1][0],
                "Huber": t[1][1],
                "Tukey": t[1][2],
                "Donut": t[1][3],
            }
        )
        testValues_01 = pd.DataFrame(
            {
                "OLS": t[2][0],
                "Huber": t[2][1],
                "Tukey": t[2][2],
                "Donut": t[2][3],
            }
        )
        testValues[k] = [testValues_1, testValues_05, testValues_01]
        k = k + 1

    # Create dataframes with C.I. results from the simulation
    ciValues = ["", ""]
    i = 0
    for c in ci_cove, ci_lenght:
        ciValues_1 = pd.DataFrame(
            {"OLS": c[0][0], "Huber": c[0][1], "Tukey": c[0][2], "Donut": c[0][3]}
        )
        ciValues_05 = pd.DataFrame(
            {"OLS": c[1][0], "Huber": c[1][1], "Tukey": c[1][2], "Donut": c[1][3]}
        )
        ciValues_01 = pd.DataFrame(
            {"OLS": c[2][0], "Huber": c[2][1], "Tukey": c[2][2], "Donut": c[2][3]}
        )
        ciValues[i] = [ciValues_1, ciValues_05, ciValues_01]
        i = i + 1

    # Return results
    return pointEstimation, testValues, ciValues, firstSample

def simulations(r, name, n, tau, alpha, beta, cutoff=0, L=0):
    """
    This method runs various simulations and return the results from all different simulations.

    Parameters
    ----------
    r: int
      The number of repetitions
    name: string
      The name of the DGP
    tau : int , Default value: 0
        The size of the treatment effect. For basic and basic linear model.
    L : int, Default value: 0
        The parameter used by Noack and Rothe to define the level of misspecification,
        L={0,10,20,30,40}.
    alpha: int, Default value: 0
        The intercept parameter of the equation. For basic linear model.
    beta: int, Default value: 0
        The slope parameter of the equation. For basic linear model.
    cutoff : int , Default value: 0
        The treshold of the running variable that determines treatment

    Returns
    -------
    simResults:Dataframe
      Object with the results from the various simulations.

    """
    # Run Simulations
    point1, test1, confInt1, firstSample1 = simulation(
        r, name, n, tau, alpha, beta, cutoff=cutoff, outlier=False
    )
    point2, test2, confInt2, firstSample2 = simulation(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        cutoff=cutoff,
        outlier=True,
        outlierMethod="Simple Outside Left",
        nOutliers=1,
    )
    point3, test3, confInt3, firstSample3 = simulation(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        cutoff=cutoff,
        outlier=True,
        outlierMethod="Simple Outside Left",
        nOutliers=3,
    )
    point4, test4, confInt4, firstSample4 = simulation(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        cutoff=cutoff,
        outlier=True,
        outlierMethod="Simple",
        nOutliers=1,
    )
    point5, test5, confInt5, firstSample5 = simulation(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        cutoff=cutoff,
        outlier=True,
        outlierMethod="Simple",
        nOutliers=3,
    )
    point6, test6, confInt6, firstSample6 = simulation(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        cutoff=cutoff,
        outlier=True,
        outlierMethod="Simple Oposite",
        nOutliers=3,
    )
    simResults = (
        point1,
        test1,
        confInt1,
        firstSample1,
        point2,
        test2,
        confInt2,
        firstSample2,
        point3,
        test3,
        confInt3,
        firstSample3,
        point4,
        test4,
        confInt4,
        firstSample4,
        point5,
        test5,
        confInt5,
        firstSample5,
        point6,
        test6,
        confInt6,
        firstSample6,
    )
    return simResults


# point6, test6, confInt6, firstSample6 = simulation(
#        r, name, n, tau=0, L=40, cutoff=cutoff, b=0.5, outlier=False
#    )
