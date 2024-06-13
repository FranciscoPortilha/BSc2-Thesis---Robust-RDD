import numpy as np
import pandas as pd

from src.exports import scenariosHist, toLatexTable
from src.rrdd import jointFitRD
from src.sample import genSample
from src.simMetrics import compJB, compRMSE


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
    Run simulation analysis for RDD: Generates a sample r times and returns the results of each sample

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
        The parameter used by NR to define the level of misspecification L={0,10,20,30,40}. For the Noack and Rothe model.
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
    sample: DataFrame
        A dataframe object with the results from the simulation.
        For each smaple it returns the estimated Treatment Effects (TE), bandwidth (H).
    """
    # Creat empty output arrays

    bias = [{}, {}, {}, {}]

    p_1 = [{}, {}, {}, {}]
    p_05 = [{}, {}, {}, {}]
    p_01 = [{}, {}, {}, {}]

    ci_1 = [{}, {}, {}, {}]
    ci_05 = [{}, {}, {}, {}]
    ci_01 = [{}, {}, {}, {}]

    models = ["OLS", "Robust Huber", "Robust Tuckey", "Donut"]
    # Generate sample and fit models r times
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

        # Estimate the different models
        for i in range(len(models)):
            # Fit the model
            res = jointFitRD(models[i], sample)

            # Record point estimate
            bias[i] = np.append(bias[i], res.params.iloc[2] - tau)

            # Calculate t test of Ho: t == tau is rejected
            if tau == 0:
                p = res.pvalues.iloc[2]
            else:
                p = res.t_test(([0, 0, 1, 0], tau)).pvalue

            # Record if t test of Ho: t == tau is rejected
            if p >= 0.1:
                p_1[i] = np.append(p_1[i], 0)
                p_05[i] = np.append(p_05[i], 0)
                p_01[i] = np.append(p_01[i], 0)

            elif p >= 0.05:
                p_1[i] = np.append(p_1[i], 1)
                p_05[i] = np.append(p_05[i], 0)
                p_01[i] = np.append(p_01[i], 0)

            elif p >= 0.01:
                p_1[i] = np.append(p_1[i], 1)
                p_05[i] = np.append(p_05[i], 1)
                p_01[i] = np.append(p_01[i], 0)

            else:
                p_1[i] = np.append(p_1[i], 1)
                p_05[i] = np.append(p_05[i], 1)
                p_01[i] = np.append(p_01[i], 1)

            # Record correct coverage of confidence interval
            if (res.conf_int(0.1)[0].iloc[2] < tau) & (
                tau < res.conf_int(0.1)[1].iloc[2]
            ):
                ci_1[i] = np.append(ci_1[i], 1)
                ci_05[i] = np.append(ci_05[i], 1)
                ci_01[i] = np.append(ci_01[i], 1)

            elif (res.conf_int()[0].iloc[2] < tau) & (tau < res.conf_int()[1].iloc[2]):
                ci_1[i] = np.append(ci_1[i], 0)
                ci_05[i] = np.append(ci_05[i], 1)
                ci_01[i] = np.append(ci_01[i], 1)

            elif (res.conf_int(0.01)[0].iloc[2] < tau) & (
                tau < res.conf_int(0.01)[1].iloc[2]
            ):
                ci_1[i] = np.append(ci_1[i], 0)
                ci_05[i] = np.append(ci_05[i], 0)
                ci_01[i] = np.append(ci_01[i], 1)
            else:
                ci_1[i] = np.append(ci_1[i], 0)
                ci_05[i] = np.append(ci_05[i], 0)
                ci_01[i] = np.append(ci_01[i], 0)

    # Adjust the format of the arrays (delete empty first cell)
    for i in range(len(models)):
        bias[i] = np.delete(bias[i], 0)

        p_1[i] = np.delete(p_1[i], 0)
        p_05[i] = np.delete(p_05[i], 0)
        p_01[i] = np.delete(p_01[i], 0)

        ci_1[i] = np.delete(ci_1[i], 0)
        ci_05[i] = np.delete(ci_05[i], 0)
        ci_01[i] = np.delete(ci_01[i], 0)

    # Create dataframes with results from the simulation
    pointEstimation = pd.DataFrame(
        {"OLS": bias[0], "Huber": bias[1], "Tukey": bias[2], "Donut": bias[3]}
    )

    testValues_1 = pd.DataFrame(
        {"OLS": p_1[0], "Huber": p_1[1], "Tukey": p_1[2], "Donut": p_1[3]}
    )
    testValues_05 = pd.DataFrame(
        {"OLS": p_05[0], "Huber": p_05[1], "Tukey": p_05[2], "Donut": p_05[3]}
    )
    testValues_01 = pd.DataFrame(
        {"OLS": p_01[0], "Huber": p_01[1], "Tukey": p_01[2], "Donut": p_01[3]}
    )

    ciCoverage_1 = pd.DataFrame(
        {"OLS": ci_1[0], "Huber": ci_1[1], "Tukey": ci_1[2], "Donut": ci_1[3]}
    )
    ciCoverage_05 = pd.DataFrame(
        {"OLS": ci_05[0], "Huber": ci_05[1], "Tukey": ci_05[2], "Donut": ci_05[3]}
    )
    ciCoverage_01 = pd.DataFrame(
        {"OLS": ci_01[0], "Huber": ci_01[1], "Tukey": ci_01[2], "Donut": ci_01[3]}
    )

    testValues = [testValues_1, testValues_05, testValues_01]
    ciCoverage = [ciCoverage_1, ciCoverage_05, ciCoverage_01]

    # Return results
    return pointEstimation, testValues, ciCoverage


def simulations(r, name, n, tau, alpha, beta, cutoff=0, L=0):
    """
    This method runs various simulations and return the results from all different simulations

    Parameters
    ----------
    r: int
      The number of repetitions
    name: string
      The name of the DGP
    tau : int , Default value: 0
        The size of the treatment effect. For basic and basic linear model.
    L : int, Default value: 0
        The parameter used by Noack and Rothe to define the level of misspecification L={0,10,20,30,40}.
    alpha: int, Default value: 0
        The intercept parameter of the equation. For basic linear model.
    beta: int, Default value: 0
        The slope parameter of the equation. For basic linear model.
    cutoff : int , Default value: 0
        The treshold of the running variable that determines treatment

    Returns
    -------
    Results:Dataframe
      Object with the means, st.var and rmse of the various simulations

    """
    # Run Simulations
    point1, test1, confInt1 = simulation(
        r, name, n, tau, alpha, beta, cutoff=cutoff, outlier=False
    )
    point2, test2, confInt2 = simulation(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        cutoff=cutoff,
        outlier=True,
        outlierMethod="Simple Outside",
        nOutliers=1,
    )
    point3, test3, confInt3 = simulation(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        cutoff=cutoff,
        outlier=True,
        outlierMethod="Simple Outside",
        nOutliers=2,
    )
    point4, test4, confInt4 = simulation(
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
    point5, test5, confInt5 = simulation(
        r,
        name,
        n,
        tau,
        alpha,
        beta,
        cutoff=cutoff,
        outlier=True,
        outlierMethod="Simple",
        nOutliers=2,
    )
    point6, test6, confInt6 = simulation(
        r, "Noack", n, tau=0, L=40, cutoff=cutoff, b=0.5, outlier=False
    )

    # Create labels for dataframe with results about Mean, St.dev and RMSE
    multiCol1a = pd.MultiIndex.from_arrays(
        [
            [
                "Scenario 1",
                "Scenario 1",
                "Scenario 1",
                "Scenario 2",
                "Scenario 2",
                "Scenario 2",
                "Scenario 3",
                "Scenario 3",
                "Scenario 3",
            ],
            [
                "Bias",
                "St.Dev.",
                "RMSE",
                "Bias",
                "St.Dev.",
                "RMSE",
                "Bias",
                "St.Dev.",
                "RMSE",
            ],
        ]
    )
    multiCol1b = pd.MultiIndex.from_arrays(
        [
            [
                "Scenario 4",
                "Scenario 4",
                "Scenario 4",
                "Scenario 5",
                "Scenario 5",
                "Scenario 5",
                "Scenario 6",
                "Scenario 6",
                "Scenario 6",
            ],
            [
                "Bias",
                "St.Dev.",
                "RMSE",
                "Bias",
                "St.Dev.",
                "RMSE",
                "Bias",
                "St.Dev.",
                "RMSE",
            ],
        ]
    )
    row = ["OLS", "Huber", "Tuckey", "Donut"]

    # Create 2 dataframes with results about Mean, St.dev and RMSE
    Results1a = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    point1.mean(),
                    point1.std(),
                    compRMSE(point1, 2),
                    point2.mean(),
                    point2.std(),
                    compRMSE(point2, 2),
                    point3.mean(),
                    point3.std(),
                    compRMSE(point3, 2),
                ]
            )
        ),
        columns=multiCol1a,
        index=row,
    )
    Results1b = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    point4.mean(),
                    point4.std(),
                    compRMSE(point4, 2),
                    point5.mean(),
                    point5.std(),
                    compRMSE(point5, 2),
                    point6.mean(),
                    point6.std(),
                    compRMSE(point6, 2),
                ]
            )
        ),
        columns=multiCol1b,
        index=row,
    )

    # Create labels for dataframe with results about JB and t-test (Normal and student-t)
    multiCol1a = pd.MultiIndex.from_arrays(
        [
            [
                "Scenario 1",
                "Scenario 1",
                "Scenario 1",
                "Scenario 2",
                "Scenario 2",
                "Scenario 2",
                "Scenario 3",
                "Scenario 3",
                "Scenario 3",
            ],
            [
                "",
                "T-Test - T1 error",
                "T-Test - T1 error",
                "",
                "T-Test - T1 error",
                "T-Test - T1 error",
                "",
                "T-Test - T1 error",
                "T-Test - T1 error",
            ],
            [
                "JB",
                "Normal",
                "Student-t",
                "JB",
                "Normal",
                "Student-t",
                "JB",
                "Normal",
                "Student-t",
            ],
        ]
    )
    multiCol1b = pd.MultiIndex.from_arrays(
        [
            [
                "Scenario 4",
                "Scenario 4",
                "Scenario 4",
                "Scenario 5",
                "Scenario 5",
                "Scenario 5",
                "Scenario 6",
                "Scenario 6",
                "Scenario 6",
            ],
            [
                "",
                "T-Test - T1 error",
                "T-Test - T1 error",
                "",
                "T-Test - T1 error",
                "T-Test - T1 error",
                "",
                "T-Test - T1 error",
                "T-Test - T1 error",
            ],
            ["JB", "N", "St-t", "JB", "N", "St-t", "JB", "N", "St-t"],
        ]
    )

    row = ["OLS", "Huber", "Tuckey", "Donut"]

    # Create 2 dataframes with results about JB and t-test (Normal and student-t) (0.1 significance)
    Results2a_1 = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    compJB(point1),
                    test1[0].mean(),
                    test1[0].mean(),
                    compJB(point2),
                    test2[0].mean(),
                    test1[0].mean(),
                    compJB(point3),
                    test3[0].mean(),
                    test1[0].mean(),
                ]
            )
        ),
        columns=multiCol1a,
        index=row,
    )

    Results2b_1 = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    compJB(point4),
                    test4[0].mean(),
                    test1[0].mean(),
                    compJB(point5),
                    test5[0].mean(),
                    test1[0].mean(),
                    compJB(point6),
                    test6[0].mean(),
                    test1[0].mean(),
                ]
            )
        ),
        columns=multiCol1b,
        index=row,
    )

    # Captions for latex tablex
    caption1 = "Bias, standard deviation and root mean squared error of the point estimates of the treatment effect"
    caption2 = ("Jarque-Bera test statistic of the simulated point estimates.",)
    "T-test incorrect rejection of $H_0: \\hat{\\tau}=\\tau$ (normal and student's-t distributions)"

    scenariosHist(
        [point1, point2, point3, point4, point5, point6],
        True,
        "images/scenariosHist.png",
    )
    toLatexTable(r, n, Results1a, Results1b, caption1, ref="table:R1")
    toLatexTable(r, n, Results2a_1, Results2b_1, caption2, ref="table:R2_1")
