"""
Code for my bachelor thesis in Econometrics and Economics, 
Outlier Robust Regression Discontinuity Designs.

Author: Francisco Portilha (479126)
"""

# Public libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# My methods. Developed for my thesis.
from src.exports import (
    plotPowerFunctionComparison,
    plotSamplesComparison,
    plotScenariosHist,
    toLatexTable,
)
from src.rrdd import jointFitRD
from src.sample import genSample
from src.simMetrics import analyseSimResults


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
    nameSample: string, Options: 'Noack', 'Basic Linear'
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
        Values are stored as point relative to the true value.
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
    point = [{}, {}, {}, {}]

    t_type1error = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]

    ci_cove = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]
    ci_lenght = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]

    models = ["OLS", "Robust Huber", "Robust Tukey", "Donut"]
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
        for m in range(len(models)):
            # Fit the model
            res = jointFitRD(models[m], sample, cutoff)

            # Record point estimates
            point[m] = np.append(point[m], res.params.iloc[2])

            # Record if t test of Ho: t == tau is rejected based on normal and student-t
            if tau == 0:
                p = res.pvalues.iloc[2]
            else:
                p = res.t_test(([0, 0, 1, 0], tau)).pvalue

            # Record if t test of Ho: t == tau is rejected
            if p >= 0.1:
                # No incorrect rejections
                t_type1error[0][m] = np.append(t_type1error[0][m], 0)
                t_type1error[1][m] = np.append(t_type1error[1][m], 0)
                t_type1error[2][m] = np.append(t_type1error[2][m], 0)
            elif p >= 0.05:
                # Incorrect rejection at 0.1 level
                t_type1error[0][m] = np.append(t_type1error[0][m], 1)
                t_type1error[1][m] = np.append(t_type1error[1][m], 0)
                t_type1error[2][m] = np.append(t_type1error[2][m], 0)
            elif p >= 0.01:
                # Incorrect rejection at 0.1 and 0.05 levels
                t_type1error[0][m] = np.append(t_type1error[0][m], 1)
                t_type1error[1][m] = np.append(t_type1error[1][m], 1)
                t_type1error[2][m] = np.append(t_type1error[2][m], 0)
            else:
                # Incorrect rejection at all levels
                t_type1error[0][m] = np.append(t_type1error[0][m], 1)
                t_type1error[1][m] = np.append(t_type1error[1][m], 1)
                t_type1error[2][m] = np.append(t_type1error[2][m], 1)
            

            # Record correct coverage of confidence interval
            if (res.conf_int(0.1)[0].iloc[2] < tau) & (
                tau < res.conf_int(0.1)[1].iloc[2]
            ):
                # Correct coverage all levels
                ci_cove[0][m] = np.append(ci_cove[0][m], 1)
                ci_cove[1][m] = np.append(ci_cove[1][m], 1)
                ci_cove[2][m] = np.append(ci_cove[2][m], 1)

            elif (res.conf_int()[0].iloc[2] < tau) & (tau < res.conf_int()[1].iloc[2]):
                # Incorrect coverage 0.1 level
                ci_cove[0][m] = np.append(ci_cove[0][m], 0)
                ci_cove[1][m] = np.append(ci_cove[1][m], 1)
                ci_cove[2][m] = np.append(ci_cove[2][m], 1)

            elif (res.conf_int(0.01)[0].iloc[2] < tau) & (
                tau < res.conf_int(0.01)[1].iloc[2]
            ):
                # Incorrect coverage 0.1 and 0.5 level
                ci_cove[0][m] = np.append(ci_cove[0][m], 0)
                ci_cove[1][m] = np.append(ci_cove[1][m], 0)
                ci_cove[2][m] = np.append(ci_cove[2][m], 1)

            else:
                # Incorrect coverage on all levels
                ci_cove[0][m] = np.append(ci_cove[0][m], 0)
                ci_cove[1][m] = np.append(ci_cove[1][m], 0)
                ci_cove[2][m] = np.append(ci_cove[2][m], 0)

            # Record lenght of C.I. for all significance levels
            ci_lenght[0][m] = np.append(
                ci_lenght[0][m],
                res.conf_int(0.1)[1].iloc[2] - res.conf_int(0.1)[0].iloc[2],
            )
            ci_lenght[1][m] = np.append(
                ci_lenght[1][m], res.conf_int()[1].iloc[2] - res.conf_int()[0].iloc[2]
            )
            ci_lenght[2][m] = np.append(
                ci_lenght[2][m],
                res.conf_int(0.01)[1].iloc[2] - res.conf_int(0.01)[0].iloc[2],
            )

    # Adjust the format of the arrays (delete empty first cell)
    for m in range(len(models)):
        point[m] = np.delete(point[m], 0)
        t_type1error[0][m] = np.delete(t_type1error[0][m], 0)
        t_type1error[1][m] = np.delete(t_type1error[1][m], 0)
        t_type1error[2][m] = np.delete(t_type1error[2][m], 0)
        for c in ci_cove, ci_lenght:
            c[0][m] = np.delete(c[0][m], 0)
            c[1][m] = np.delete(c[1][m], 0)
            c[2][m] = np.delete(c[2][m], 0)

    # Create dataframe with point estimation results from the simulation
    pointEstimation = pd.DataFrame(
        {"OLS": point[0], "Huber": point[1], "Tukey": point[2], "Donut": point[3]}
    )

    # Create dataframes with t-test results from the simulation
    testValues_1 = pd.DataFrame(
        {
            "OLS":   t_type1error[0][0],
            "Huber": t_type1error[0][1],
            "Tukey": t_type1error[0][2],
            "Donut": t_type1error[0][3],
        }
    )
    testValues_05 = pd.DataFrame(
        {
            "OLS":   t_type1error[1][0],
            "Huber": t_type1error[1][1],
            "Tukey": t_type1error[1][2],
            "Donut": t_type1error[1][3],
        }
    )
    testValues_01 = pd.DataFrame(
        {
            "OLS":   t_type1error[2][0],
            "Huber": t_type1error[2][1],
            "Tukey": t_type1error[2][2],
            "Donut": t_type1error[2][3],
        }
    )
    testValues = [testValues_1, testValues_05, testValues_01]

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


def simulations(r, name, n, tau, alpha, beta, cutoff=0, L=0, parametersScenarios=""):
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
    (
        scenario2_method,
        scenario2_num,
        scenario3_method,
        scenario3_num,
        scenario4_method,
        scenario4_num,
        scenario5_method,
        scenario5_num,
        scenario6_method,
        scenario6_num,
    ) = parametersScenarios
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
        outlierMethod=scenario2_method,
        nOutliers=scenario2_num,
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
        outlierMethod=scenario3_method,
        nOutliers=scenario3_num,
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
        outlierMethod=scenario4_method,
        nOutliers=scenario4_num,
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
        outlierMethod=scenario5_method,
        nOutliers=scenario5_num,
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
        outlierMethod=scenario6_method,
        nOutliers=scenario6_num,
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
    
    analyseSimResults(simResults, tau, printToLatex=True)


def powerSimulation(
    taus,
    r,
    nameSample,
    n,
    alpha,
    beta,
    cutoff,
    L=0,
    b=1,
    outlier=False,
    outlierMethod="",
    nOutliers=0,
):

    rejectionRate = [{}, {}, {}, {}]
    models = ["OLS", "Robust Huber", "Robust Tukey", "Donut"]

    for i in range(17):
        t_rejections = [{}, {}, {}, {}]
        for k in range(r):
            # Generate a new sample
            sample = genSample(
                nameSample,
                n,
                taus[i],
                alpha,
                beta,
                L,
                cutoff,
                outlier,
                outlierMethod,
                nOutliers,
                False,
            )
            for m in range(len(models)):
                res = jointFitRD(models[m], sample, cutoff)
                p = res.pvalues.iloc[2]
                if p >= 0.05:
                    t_rejections[m] = np.append(t_rejections[m], 0)
                else:
                    t_rejections[m] = np.append(t_rejections[m], 1)

        for m in range(len(models)):
            t_rejections[m] = np.delete(t_rejections[m], 0)
            rejectionRate[m] = np.append(rejectionRate[m], t_rejections[m].mean())

    for m in range(len(models)):
        rejectionRate[m] = np.delete(rejectionRate[m], 0)

    return rejectionRate


def powerSimulations(
    r,
    nameSample,
    n,
    alpha,
    beta,
    cutoff,
    parametersScenarios,
):
    (
        scenario2_method,
        scenario2_num,
        scenario3_method,
        scenario3_num,
        scenario4_method,
        scenario4_num,
        scenario5_method,
        scenario5_num,
        scenario6_method,
        scenario6_num,
    ) = parametersScenarios
    taus = [{}]
    for i in range(-8, 9):
        taus = np.append(taus, i / 4)
    taus = np.delete(taus, 0)

    rejectionRates = (
        powerSimulation(taus, r, nameSample, n, alpha, beta, cutoff=0),
        powerSimulation(
            taus,
            r,
            nameSample,
            n,
            alpha,
            beta,
            cutoff,
            outlier=True,
            outlierMethod=scenario2_method,
            nOutliers=scenario2_num,
        ),
        powerSimulation(
            taus,
            r,
            nameSample,
            n,
            alpha,
            beta,
            cutoff,
            outlier=True,
            outlierMethod=scenario3_method,
            nOutliers=scenario3_num,
        ),
        powerSimulation(
            taus,
            r,
            nameSample,
            n,
            alpha,
            beta,
            cutoff,
            outlier=True,
            outlierMethod=scenario4_method,
            nOutliers=scenario4_num,
        ),
        powerSimulation(
            taus,
            r,
            nameSample,
            n,
            alpha,
            beta,
            cutoff,
            outlier=True,
            outlierMethod=scenario5_method,
            nOutliers=scenario5_num,
        ),
        powerSimulation(
            taus,
            r,
            nameSample,
            n,
            alpha,
            beta,
            cutoff,
            outlier=True,
            outlierMethod=scenario6_method,
            nOutliers=scenario6_num,
        ),
    )

    plotPowerFunctionComparison(
        taus,
        rejectionRates,
        True,
        "images/powerFunctions.png",
    )
