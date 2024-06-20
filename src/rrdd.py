"""
Code for my bachelor thesis in Econometrics and Economics, 
Outlier Robust Regression Discontinuity Designs.

Author: Francisco Portilha (479126)
"""

# Public libraries
import numpy as np
import statsmodels.api as sm


def prepExog(sample, intercept=False, jointFit=False, outliers=True):
    """
    This method add prepares a sample to be used as exogenous variables in regression.

    Parameters
    ----------
    sample : DataFrame
        The sample to prepare.
    intercept : boolean
        Determines if a intercept is added to the exogenous variables.

    Returns
    -------
    exog : DataFrame
        Object with the prepared exogenous variables to be used for the regression.
    """
    # Copy object and delete columns that are not exogenous variables.
    exog = sample.copy()
    exog = exog.drop("Y", axis="columns")
    if jointFit is False:
        exog = exog.drop("Treatment", axis="columns")
    if outliers:  
        exog = exog.drop("Outlier", axis="columns")
    # Add an intercept if requested
    if intercept == True:
        exog = sm.add_constant(exog)

    return exog


def fit(name, sample, intercept, cutoff=0, jointFit=False, outliers=True, donut=0.1):
    """
    This method will fit a regression based on the different estimation methods
    for the given sample.

    Parameters
    ----------
    name : string
        The name of the estimation method to use.
        Opiton values : Robust Huber, Robust Tukey, OLS, Donut.
    sample : DataFrame
        The sample to estimate the regression for.
    intercept : boolean
        Determines if an intercept is added to the sample.
    cutoff : int
        The value of the threshold in the running variable.

    Returns
    -------
    res.params : object
        The parameters of the regression.
    """

    # Estimate regression based on estimation method
    if name == "Robust Huber":
        exog = prepExog(sample, intercept, jointFit, outliers)
        res = sm.RLM(sample.Y, exog, M=sm.robust.norms.HuberT())

    elif name == "Robust Tukey":
        exog = prepExog(sample, intercept, jointFit, outliers)
        res = sm.RLM(sample.Y, exog, M=sm.robust.norms.TukeyBiweight())

    elif name == "OLS":
        exog = prepExog(sample, intercept, jointFit, outliers)
        res = sm.OLS(sample.Y, exog)

    elif name == "Donut":
        sample = sample.loc[np.abs(sample.X - cutoff) >= donut]
        exog = prepExog(sample, intercept, jointFit, outliers)
        res = sm.OLS(sample.Y, exog)

    else:
        return NameError("Type of Estimation method is not recognised")
    # Fit model and return parameters
    res = res.fit()
    return res


def splitFitRD(name, sample, cutoff=0):
    """
    This method estimates the treatment effects based on RDD estimated by 2 regression
    on eeach side of the cutoff.

    Parameters
    ----------
    name : string
        The name of the estimation method to use.
        Opiton values : Robust Huber, Robust Tukey, OLS, Donut.
    sample : DataFrame
        The sample to estimate the regression for.
    cutoff : int
        The value of the threshold in the running variable.

    Returns
    -------
    tau : int
        The estimated treatment effect.
    """
    # Split sample at cutoff
    sample_below = sample.loc[sample.X <= cutoff]
    sample_above = sample.loc[sample.X > cutoff]
    params_below = fit(name, sample_below, intercept=True).params
    params_above = fit(name, sample_above, intercept=True).params
    tau = params_above.iloc[0] - params_below.iloc[0]
    return tau


def jointFitRD(name, sample, cutoff=0, b=1,outliers=True,donut=0.1):
    # Create new column with X*T covariate
    sample["XT"] = sample.X * sample.Treatment

    # Select part of the sample to be used according to bandwidth
    sample = sample.loc[np.abs(sample.X - cutoff) <= b]

    # fit model and return result
    return fit(name, sample, True, cutoff, True, outliers, donut)
