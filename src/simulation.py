"""
Code for my bachelor thesis in Econometrics and Economics, 
Outlier Robust Regression Discontinuity Designs.

Author: Francisco Portilha (479126)
"""

# Public libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# My methods. Developed for my thesis.
from src.exports import (
    plotAsymptoticComparison,
    plotPowerFunctionComparison,
    plotSamplesComparison,
    plotScenariosHist,
    toLatexTable,
)
from src.rrdd import jointFitRD
from src.sample import genSample
from src.simMetrics import analyseSimResults


def powerSimulation(
    r,
    taus,
    specialTau,
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
    computeAsymptotics=False,
):
    rejectionRate = [{}, {}, {}, {}]
    labels = ["OLS", "Huber", "Tukey", "Donut"]
    detailedResults = [{}, {}]

    k = 0
    for t in range(len(taus)):
        # If it is a tau of special interest do a simulation with detailed results
        if taus[t] in specialTau:
            if computeAsymptotics:
                detailedResults[k] = asymptoticSimulation(
                    r,
                    nameSample,
                    n,
                    taus[t],
                    alpha,
                    beta,
                    cutoff=cutoff,
                    L=L,
                    b=b,
                    outlier=outlier,
                    outlierMethod=outlierMethod,
                    nOutliers=nOutliers,
                )
            else:
                detailedResults[k] = simulationDetailed(
                    r,
                    nameSample,
                    n,
                    taus[t],
                    alpha,
                    beta,
                    cutoff=cutoff,
                    L=L,
                    b=b,
                    outlier=outlier,
                    outlierMethod=outlierMethod,
                    nOutliers=nOutliers,
                )
            for m in range(4):
                rejectionRate[m] = np.append(
                    rejectionRate[m], detailedResults[k][1][1][1][labels[m]].mean()
                )
            k = k + 1

        # If tau is not of special interest do a simulation that only record the power of t-test
        else:
            t_rejections = simulationShort(
                taus[t],
                r,
                nameSample,
                n,
                alpha,
                beta,
                cutoff,
                L=L,
                b=b,
                outlier=outlier,
                outlierMethod=outlierMethod,
                nOutliers=nOutliers,
            )
            for m in range(4):
                rejectionRate[m] = np.append(rejectionRate[m], t_rejections[m].mean())

    for m in range(4):
        rejectionRate[m] = np.delete(rejectionRate[m], 0)

    return (
        rejectionRate,
        detailedResults,
    )


def asymptoticSimulation(
    r,
    nameSample,
    nInit,
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
    nRange = {}
    labels = ["OLS", "Huber", "Tukey", "Donut"]
    bias, stDev, rmse, effi, ciCc, ciSize, t1Error, t2Error = (
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
    )
    for k in range(12):
        n = int(nInit * np.power(1.6, k - 3))
        nRange = np.append(nRange, n)
        pointEstimation_n, testValues_n, ciValues_n, firstSample_n, notUsed = (
            simulationDetailed(
                r,
                nameSample,
                n,
                tau,
                alpha,
                beta,
                L,
                cutoff,
                b,
                outlier,
                outlierMethod,
                nOutliers,
            )
        )
        if k == 3:
            pointEstimation, testValues, ciValues, firstSample = (
                pointEstimation_n,
                testValues_n,
                ciValues_n,
                firstSample_n,
            )
        for m in range(4):
            bias[m] = np.append(bias[m], pointEstimation_n[labels[m]].mean() - tau)
            stDev[m] = np.append(stDev[m], pointEstimation_n[labels[m]].std())
            rmse[m] = np.append(
                rmse[m], sm.tools.eval_measures.rmse(pointEstimation_n[labels[m]], tau)
            )
            effi[m] = np.append(
                effi[m],
                pointEstimation_n["Tukey"].std() / pointEstimation_n[labels[m]].std(),
            )
            ciCc[m] = np.append(ciCc[m], ciValues_n[0][1][labels[m]].mean())
            ciSize[m] = np.append(ciSize[m], ciValues_n[1][1][labels[m]].mean())
            t1Error[m] = np.append(t1Error[m], testValues_n[0][1][labels[m]].mean())
            t2Error[m] = np.append(t2Error[m], testValues_n[1][1][labels[m]].mean())

    for metric in bias, stDev, rmse, effi, ciCc, ciSize, t1Error, t2Error:
        for m in range(4):
            metric[m] = np.delete(metric[m], 0)
    nRange = np.delete(nRange, 0)

    return (
        pointEstimation,
        testValues,
        ciValues,
        firstSample,
        (bias, stDev, rmse, effi, ciCc, ciSize, t1Error, t2Error, nRange),
    )


def simulationShort(
    tau,
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
    models = ["OLS", "Robust Huber", "Robust Tukey", "Donut"]
    t_rejections = [{}, {}, {}, {}]
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

        for m in range(len(models)):
            res = jointFitRD(models[m], sample, cutoff)
            p = res.pvalues.iloc[2]
            if p >= 0.05:
                t_rejections[m] = np.append(t_rejections[m], 0)
            else:
                t_rejections[m] = np.append(t_rejections[m], 1)

    for m in range(len(models)):
        t_rejections[m] = np.delete(t_rejections[m], 0)

    return t_rejections


def simulationDetailed(
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

    ci_cove = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]
    ci_lenght = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]

    t_type1error = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]
    t_type2error = [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]

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
            # Calculate if t test statistic of Ho: t == tau
            if tau == 0:
                p_t1 = res.pvalues.iloc[2]
            else:
                p_t1 = res.t_test(([0, 0, 1, 0], tau)).pvalue

            # Record if t test of Ho: t == tau is rejected
            if p_t1 >= 0.1:
                # No incorrect rejections
                t_type1error[0][m] = np.append(t_type1error[0][m], 0)
                t_type1error[1][m] = np.append(t_type1error[1][m], 0)
                t_type1error[2][m] = np.append(t_type1error[2][m], 0)
            elif p_t1 >= 0.05:
                # Incorrect rejection at 0.1 level
                t_type1error[0][m] = np.append(t_type1error[0][m], 1)
                t_type1error[1][m] = np.append(t_type1error[1][m], 0)
                t_type1error[2][m] = np.append(t_type1error[2][m], 0)
            elif p_t1 >= 0.01:
                # Incorrect rejection at 0.1 and 0.05 levels
                t_type1error[0][m] = np.append(t_type1error[0][m], 1)
                t_type1error[1][m] = np.append(t_type1error[1][m], 1)
                t_type1error[2][m] = np.append(t_type1error[2][m], 0)
            else:
                # Incorrect rejection at all levels
                t_type1error[0][m] = np.append(t_type1error[0][m], 1)
                t_type1error[1][m] = np.append(t_type1error[1][m], 1)
                t_type1error[2][m] = np.append(t_type1error[2][m], 1)

            # Calculate if t test statistic of Ho: t == 0
            p_t2 = res.pvalues.iloc[2]

            # Record if t test of Ho: t == 0 is rejected
            if p_t2 >= 0.1:
                # No incorrect rejections
                t_type2error[0][m] = np.append(t_type2error[0][m], 0)
                t_type2error[1][m] = np.append(t_type2error[1][m], 0)
                t_type2error[2][m] = np.append(t_type2error[2][m], 0)
            elif p_t2 >= 0.05:
                # Incorrect rejection at 0.1 level
                t_type2error[0][m] = np.append(t_type2error[0][m], 1)
                t_type2error[1][m] = np.append(t_type2error[1][m], 0)
                t_type2error[2][m] = np.append(t_type2error[2][m], 0)
            elif p_t2 >= 0.01:
                # Incorrect rejection at 0.1 and 0.05 levels
                t_type2error[0][m] = np.append(t_type2error[0][m], 1)
                t_type2error[1][m] = np.append(t_type2error[1][m], 1)
                t_type2error[2][m] = np.append(t_type2error[2][m], 0)
            else:
                # Incorrect rejection at all levels
                t_type2error[0][m] = np.append(t_type2error[0][m], 1)
                t_type2error[1][m] = np.append(t_type2error[1][m], 1)
                t_type2error[2][m] = np.append(t_type2error[2][m], 1)

    # Adjust the format of the arrays (delete empty first cell)
    for m in range(len(models)):
        point[m] = np.delete(point[m], 0)
        for t in t_type1error, t_type2error:
            t[0][m] = np.delete(t[0][m], 0)
            t[1][m] = np.delete(t[1][m], 0)
            t[2][m] = np.delete(t[2][m], 0)
        for c in ci_cove, ci_lenght:
            c[0][m] = np.delete(c[0][m], 0)
            c[1][m] = np.delete(c[1][m], 0)
            c[2][m] = np.delete(c[2][m], 0)

    # Create dataframe with point estimation results from the simulation
    pointEstimation = pd.DataFrame(
        {"OLS": point[0], "Huber": point[1], "Tukey": point[2], "Donut": point[3]}
    )

    # Create dataframes with t-test results from the simulation
    testValues = ["", ""]
    i = 0
    for t in t_type1error, t_type2error:
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
        testValues[i] = [testValues_1, testValues_05, testValues_01]
        i = i + 1

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
    return pointEstimation, testValues, ciValues, firstSample, {}


def simulations(r, name, n, tau=0, alpha=0, beta=0, cutoff=0, L=[0,0,0,0,0,0], b=1, parametersScenarios=["","","","","","","","","",""], printToLatex=False):
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
    point1, test1, confInt1, firstSample1, notused = simulationDetailed(
        r, name, n, tau, alpha, beta, L[0], cutoff=cutoff, b=b, outlier=False
    )
    point2, test2, confInt2, firstSample2, notused = simulationDetailed(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        L[1],
        cutoff=cutoff,
        b=b,
        outlier=True,
        outlierMethod=scenario2_method,
        nOutliers=scenario2_num,
    )
    point3, test3, confInt3, firstSample3, notused = simulationDetailed(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        L[2],
        cutoff=cutoff,
        b=b,
        outlier=True,
        outlierMethod=scenario3_method,
        nOutliers=scenario3_num,
    )
    point4, test4, confInt4, firstSample4, notused = simulationDetailed(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        L[3],
        cutoff=cutoff,
        b=b,
        outlier=True,
        outlierMethod=scenario4_method,
        nOutliers=scenario4_num,
    )
    point5, test5, confInt5, firstSample5, notused= simulationDetailed(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        L[4],
        cutoff=cutoff,
        b=b,
        outlier=True,
        outlierMethod=scenario5_method,
        nOutliers=scenario5_num,
    )
    #if name=="Noack":
    #    point6, test6, confInt6, firstSample6 = point1, test1, confInt1, firstSample1
    #
    #else:
    point6, test6, confInt6, firstSample6, notused = simulationDetailed(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        L[5],
        cutoff=cutoff,
        b=b,
        outlier=True,
        outlierMethod=scenario6_method,
        nOutliers=scenario6_num,
    )
    simResults = (
        firstSample1,
        firstSample2,
        firstSample3,
        firstSample4,
        firstSample5,
        firstSample6,
        point1,
        point2,
        point3,
        point4,
        point5,
        point6,
        confInt1,
        confInt2,
        confInt3,
        confInt4,
        confInt5,
        confInt6,
        test1,
        test2,
        test3,
        test4,
        test5,
        test6,
    )

    analyseSimResults(simResults, tau, printToLatex)


def powerSimulations(
    r,
    nameSample,
    n,
    alpha,
    beta,
    cutoff=0,
    parametersScenarios="",
    specialTau=0,
    computeAsymptotics=False,
    prinToLatex=False,
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

    rejectionRate1, detailedResults1 = powerSimulation(
        r,
        taus,
        specialTau,
        nameSample,
        n,
        alpha,
        beta,
        cutoff=0,
        computeAsymptotics=computeAsymptotics,
    )

    rejectionRate2, detailedResults2 = powerSimulation(
        r,
        taus,
        specialTau,
        nameSample,
        n,
        alpha,
        beta,
        cutoff,
        outlier=True,
        outlierMethod=scenario2_method,
        nOutliers=scenario2_num,
        computeAsymptotics=computeAsymptotics, 

    )

    rejectionRate3, detailedResults3 = powerSimulation(
        r,
        taus,
        specialTau,
        nameSample,
        n,
        alpha,
        beta,
        cutoff,
        outlier=True,
        outlierMethod=scenario3_method,
        nOutliers=scenario3_num,
        computeAsymptotics=computeAsymptotics,
    )

    rejectionRate4, detailedResults4 = powerSimulation(
        r,
        taus,
        specialTau,
        nameSample,
        n,
        alpha,
        beta,
        cutoff,
        outlier=True,
        outlierMethod=scenario4_method,
        nOutliers=scenario4_num,
        computeAsymptotics=computeAsymptotics,
    )

    rejectionRate5, detailedResults5 = powerSimulation(
        r,
        taus,
        specialTau,
        nameSample,
        n,
        alpha,
        beta,
        cutoff,
        outlier=True,
        outlierMethod=scenario5_method,
        nOutliers=scenario5_num,
        computeAsymptotics=computeAsymptotics,
    )

    rejectionRate6, detailedResults6 = powerSimulation(
        r,
        taus,
        specialTau,
        nameSample,
        n,
        alpha,
        beta,
        cutoff,
        outlier=True,
        outlierMethod=scenario6_method,
        nOutliers=scenario6_num,
        computeAsymptotics=computeAsymptotics,
    )

    for t in range(len(specialTau)):
        (pointEstim1, testValues1, ciValues1, firstSample1, asymp1) = detailedResults1[
            t
        ]
        (pointEstim2, testValues2, ciValues2, firstSample2, asymp2) = detailedResults2[
            t
        ]
        (pointEstim3, testValues3, ciValues3, firstSample3, asymp3) = detailedResults3[
            t
        ]
        (pointEstim4, testValues4, ciValues4, firstSample4, asymp4) = detailedResults4[
            t
        ]
        (pointEstim5, testValues5, ciValues5, firstSample5, asymp5) = detailedResults5[
            t
        ]
        (pointEstim6, testValues6, ciValues6, firstSample6, asymp6) = detailedResults6[
            t
        ]
        simResults = (
            firstSample1,
            firstSample2,
            firstSample3,
            firstSample4,
            firstSample5,
            firstSample6,
            pointEstim1,
            pointEstim2,
            pointEstim3,
            pointEstim4,
            pointEstim5,
            pointEstim6,
            ciValues1,
            ciValues2,
            ciValues3,
            ciValues4,
            ciValues5,
            ciValues6,
            testValues1,
            testValues2,
            testValues3,
            testValues4,
            testValues5,
            testValues6,
        )

        analyseSimResults(simResults, specialTau[t], prinToLatex)
        if computeAsymptotics:
            plotAsymptoticComparison(
                specialTau[t],
                (
                    asymp1,
                    asymp2,
                    asymp3,
                    asymp4,
                    asymp5,
                    asymp6,
                ),
                True,
                "images/asymptotic",
            )

    plotPowerFunctionComparison(
        taus,
        (
            rejectionRate1,
            rejectionRate2,
            rejectionRate3,
            rejectionRate4,
            rejectionRate5,
            rejectionRate6,
        ),
        True,
        "images/powerFunctions.png",
    )
