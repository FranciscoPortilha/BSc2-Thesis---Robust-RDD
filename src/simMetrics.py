"""
Code for my bachelor thesis in Econometrics and Economics, 
Outlier Robust Regression Discontinuity Designs.

Author: Francisco Portilha (479126)
"""

# Public libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.stattools as st

# My methods. Developed for my thesis.
from src.exports import plotSamplesComparison, plotScenariosHist, toLatexTable


def compRMSE(pointEstimation, trueValue):
    """
    This method computes the RMSE of the estimates for all estimation methods.
    """
    return [
        sm.tools.eval_measures.rmse(pointEstimation.OLS, trueValue),
        sm.tools.eval_measures.rmse(pointEstimation.Huber, trueValue),
        sm.tools.eval_measures.rmse(pointEstimation.Tukey, trueValue),
        sm.tools.eval_measures.rmse(pointEstimation.Donut, trueValue),
    ]


def compSkew(pointEstimation):
    """
    This method computes the skweness of the point estimation for all estimation mathods.
    """

    return [
        st.jarque_bera(pointEstimation.OLS)[2],
        st.jarque_bera(pointEstimation.Huber)[2],
        st.jarque_bera(pointEstimation.Tukey)[2],
        st.jarque_bera(pointEstimation.Donut)[2],
    ]


def compKurt(pointEstimation):
    """
    This method computes the skweness of the point estimation for all estimation mathods.
    """

    return [
        st.jarque_bera(pointEstimation.OLS)[3],
        st.jarque_bera(pointEstimation.Huber)[3],
        st.jarque_bera(pointEstimation.Tukey)[3],
        st.jarque_bera(pointEstimation.Donut)[3],
    ]


def compJB(pointEstimation):
    """
    This method computes the jarque-bera test for all estimation mathods.
    """

    return [
        st.jarque_bera(pointEstimation.OLS)[1],
        st.jarque_bera(pointEstimation.Huber)[1],
        st.jarque_bera(pointEstimation.Tukey)[1],
        st.jarque_bera(pointEstimation.Donut)[1],
    ]


def percentV(values):
    return pd.DataFrame(
        {"0.1": values[0].mean(), "0.05": values[1].mean(), "0.01": values[2].mean()}
    )


def analyseSimResults(simResults, tau, b=1, printToLatex=False, figureFolder=""):
    """
    This method analyses the results from the simulation, saves the various figures,
    prints the latex tables.

    Parameters
    ----------
    simResults: Dataframe
        Object with all the results from simulations method
    """
    (
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
    ) = simResults
    # Create lables that are used in multiple tables
    labelsScenariosA_2 = [
        "Scenario 1",
        "Scenario 1",
        "Scenario 2",
        "Scenario 2",
        "Scenario 3",
        "Scenario 3",
    ]
    labelsScenariosB_2 = [
        "Scenario 4",
        "Scenario 4",
        "Scenario 5",
        "Scenario 5",
        "Scenario 6",
        "Scenario 6",
    ]
    labelsScenariosA = [
        "Scenario 1",
        "Scenario 1",
        "Scenario 1",
        "Scenario 2",
        "Scenario 2",
        "Scenario 2",
        "Scenario 3",
        "Scenario 3",
        "Scenario 3",
    ]
    labelsScenariosB = [
        "Scenario 4",
        "Scenario 4",
        "Scenario 4",
        "Scenario 5",
        "Scenario 5",
        "Scenario 5",
        "Scenario 6",
        "Scenario 6",
        "Scenario 6",
    ]
    labelsScenariosA_4 = [
        "Scenario 1",
        "Scenario 1",
        "Scenario 1",
        "Scenario 1",
        "Scenario 2",
        "Scenario 2",
        "Scenario 2",
        "Scenario 2",
        "Scenario 3",
        "Scenario 3",
        "Scenario 3",
        "Scenario 3",
    ]
    labelsScenariosB_4 = [
        "Scenario 4",
        "Scenario 4",
        "Scenario 4",
        "Scenario 4",
        "Scenario 5",
        "Scenario 5",
        "Scenario 5",
        "Scenario 5",
        "Scenario 6",
        "Scenario 6",
        "Scenario 6",
        "Scenario 6",
    ]
    row = ["OLS", "Huber", "Tukey", "Donut"]

    # Create labels for dataframe with results about Mean, St.dev and RMSE
    labelsResults1 = [
        "Bias",
        "St.Dev.",
        "RMSE",
        "Bias",
        "St.Dev.",
        "RMSE",
        "Bias",
        "St.Dev.",
        "RMSE",
    ]
    labelsResults1a = pd.MultiIndex.from_arrays(
        [
            labelsScenariosA,
            labelsResults1,
        ]
    )
    labelsResults1b = pd.MultiIndex.from_arrays(
        [
            labelsScenariosB,
            labelsResults1,
        ]
    )

    # Create 2 dataframes with results about Mean, St.dev and RMSE
    Results1a = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    point1.mean() - tau,
                    point1.std(),
                    compRMSE(point1, tau),
                    point2.mean() - tau,
                    point2.std(),
                    compRMSE(point2, tau),
                    point3.mean() - tau,
                    point3.std(),
                    compRMSE(point3, tau),
                ]
            )
        ),
        columns=labelsResults1a,
        index=row,
    )
    Results1b = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    point4.mean() - tau,
                    point4.std(),
                    compRMSE(point4, tau),
                    point5.mean() - tau,
                    point5.std(),
                    compRMSE(point5, tau),
                    point6.mean() - tau,
                    point6.std(),
                    compRMSE(point6, tau),
                ]
            )
        ),
        columns=labelsResults1b,
        index=row,
    )

    # Create labels for dataframe with results about skewness kurtosis and JB test
    labelsResults2 = [
        "Skew",
        "Kurt",
        "JB",
        "Skew",
        "Kurt",
        "JB",
        "Skew",
        "Kurt",
        "JB",
    ]
    labelsResults2a = pd.MultiIndex.from_arrays(
        [
            labelsScenariosA,
            labelsResults2,
        ]
    )
    labelsResults2b = pd.MultiIndex.from_arrays(
        [
            labelsScenariosB,
            labelsResults2,
        ]
    )

    # Create 2 dataframes with results about skewness kurtosis and JB test
    Results2a = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    compSkew(point1),
                    compKurt(point1),
                    compJB(point1),
                    compSkew(point2),
                    compKurt(point2),
                    compJB(point2),
                    compSkew(point3),
                    compKurt(point3),
                    compJB(point3),
                ]
            )
        ),
        columns=labelsResults2a,
        index=row,
    )
    Results2b = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    compSkew(point4),
                    compKurt(point4),
                    compJB(point4),
                    compSkew(point5),
                    compKurt(point5),
                    compJB(point5),
                    compSkew(point6),
                    compKurt(point6),
                    compJB(point6),
                ]
            )
        ),
        columns=labelsResults2b,
        index=row,
    )

    # Create labels for dataframe about correct coverage of C.I. and lenght
    labelsResults3 = [
        "C.C.",
        "Length",
        "C.C.",
        "Length",
        "C.C.",
        "Length",
    ]
    labelsResults3a = pd.MultiIndex.from_arrays(
        [
            labelsScenariosA_2,
            labelsResults3,
        ]
    )
    labelsResults3b = pd.MultiIndex.from_arrays(
        [
            labelsScenariosB_2,
            labelsResults3,
        ]
    )

    # Create 2 dataframes with results about correct coverage of C.I. and lenght
    Results3a = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    confInt1[0][1].mean(),
                    confInt1[1][1].mean(),
                    confInt2[0][1].mean(),
                    confInt2[1][1].mean(),
                    confInt3[0][1].mean(),
                    confInt3[1][1].mean(),
                ]
            )
        ),
        columns=labelsResults3a,
        index=row,
    )
    Results3b = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    confInt4[0][1].mean(),
                    confInt4[1][1].mean(),
                    confInt5[0][1].mean(),
                    confInt5[1][1].mean(),
                    confInt6[0][1].mean(),
                    confInt6[1][1].mean(),
                ]
            )
        ),
        columns=labelsResults3b,
        index=row,
    )

    # Create labels for dataframe with results about type 1 error of t-test
    labelsResults4 = [
        "T.I",
        "T.II",
        "T.I",
        "T.II",
        "T.I",
        "T.II",
    ]
    labelsResults4a = pd.MultiIndex.from_arrays(
        [
            labelsScenariosA_2,
            labelsResults4,
        ]
    )
    labelsResults4b = pd.MultiIndex.from_arrays(
        [
            labelsScenariosB_2,
            labelsResults4,
        ]
    )

    # Create 2 dataframes with results about type 1 error of t-test
    Results4a = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    test1[0][1].mean(),
                    test1[1][1].mean(),
                    test2[0][1].mean(),
                    test2[1][1].mean(),
                    test3[0][1].mean(),
                    test3[1][1].mean(),
                ]
            )
        ),
        columns=labelsResults4a,
        index=row,
    )
    Results4b = pd.DataFrame(
        np.transpose(
            np.array(
                [
                    test4[0][1].mean(),
                    test4[1][1].mean(),
                    test5[0][1].mean(),
                    test5[1][1].mean(),
                    test6[0][1].mean(),
                    test6[1][1].mean(),
                ]
            )
        ),
        columns=labelsResults4b,
        index=row,
    )

    # Captions for latex tablex
    caption1 = (
        "Bias, standard deviation and root mean squared error of the point estimates "
        + "of the treatment effect. From the simulation with $r = "
        + str(len(point1))
        + "$ , $\\tau = "
        + str(tau)
        + "$."
    )
    caption2 = (
        "Skewness, kurtosis and p-value of the jarque-bera statistic of the estimated treatment effects. "
        + "From the simulation with $r = "
        + str(len(point1))
        + "$ , $\\tau = "
        + str(tau)
        + "$."
    )
    caption3 = (
        "Correct coverage of the confidence intervals for $\\tau$ and their length. For significance level of $\\alpha=0.05$. "
        + "From the simulation with $r = "
        + str(len(point1))
        + "$ , $\\tau = "
        + str(tau)
        + "$."
    )
    caption4 = (
        "Type I and type II errors of t-test for $h_0:\\hat{\\tau}=\\tau$ and $h_0:\\hat{\\tau}=0$ repectively, "
        + "from the estimated treatment effects. From the simulation with $r = "
        + str(len(point1))
        + "$ , $\tau = "
        + str(tau)
        + "$. For significance level of $\\alpha=0.05$."
    )
    captions = caption1, caption2, caption3, caption4
    # Plot figures and print latex tables
    plotSamplesComparison(
        [
            firstSample1,
            firstSample2,
            firstSample3,
            firstSample4,
            firstSample5,
            firstSample6,
        ],
        True,
        "images/" + figureFolder + "/sampleComparison_tau_" + str(tau) + ".png",
        b=b,
    )
    plotSamplesComparison(
        [
            firstSample1,
            firstSample2,
            firstSample3,
            firstSample4,
            firstSample5,
            firstSample6,
        ],
        True,
        "images/" + figureFolder + "/regressionComparison_tau_" + str(tau) + ".png",
        True,
        b=b,
    )
    plotScenariosHist(
        [point1, point2, point3, point4, point5, point6],
        tau,
        True,
        "images/" + figureFolder + "/scenariosHist_tau_" + str(tau) + ".png",
    )
    print("")
    print(
        "------------- Results for: n = "
        + str(len(firstSample1.X))
        + " , r = "
        + str(len(point1.OLS))
        + " , τ = "
        + str(tau)
        + " -------------"
    )
    print("")

    if printToLatex:
        print("------------- Results in Latex Code -------------")
        print("")
        toLatexTable(Results1a, Results1b, caption1, ref="table:R1")
        toLatexTable(Results2a, Results2b, caption2, ref="table:R2")
        toLatexTable(Results3a, Results3b, caption3, ref="table:R3")
        toLatexTable(Results4a, Results4b, caption4, ref="table:R4")
    print("------------- Results -------------")
    print("")
    i = 0
    for table in (
        (Results1a, Results1b),
        (Results2a, Results2b),
        (Results3a, Results3b),
        (Results4a, Results4b),
    ):
        print("")
        print("Table " + str(1 + i) + " - " + captions[i])
        print("")
        print(table[0])
        print(table[1])
        i = i + 1
    print("")
    print("------------- Thank you for running my thesis project -------------")
    print("")
