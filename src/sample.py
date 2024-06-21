"""
Code for my bachelor thesis in Econometrics and Economics, 
Outlier Robust Regression Discontinuity Designs.

Author: Francisco Portilha (479126)
"""

# Public libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sign(x):
    """
    This function computes the sign of the given observation.

    Parameters
    ----------
    x : int
        The observation to compute the sign.

    Returns
    -------
    sign: int
        Returns -1 if x is negative and 1 if nonnegative.
    """
    if x < 0:
        return -1
    else:
        return 1


def treatment(x,cutoff=0,positive=True):
    """
    This function computes if an observation has received treatment.

    Parameters
    ----------
    x : int
        The observation to compute the treatment variable.

    Returns
    -------
    treatment: int
        Returns 0 if observation x does not receive treatment and 1 if it does receive treatment.
    """
    if positive:
        if x < cutoff:
            return 0
        else:
            return 1
    else:
        if x > cutoff:
            return 0
        else:
            return 1


def indicator(x):
    """
    This indicator function computes if an observation is within a distance of the threshold.

    Parameters
    ----------
    x : int
        The observation to compute the indicator value.

    Returns
    -------
    treatment: int
        Returns 1 if observation x is within the distance and 0 if it is not.
    """
    if np.abs(x) < 0.1:
        return 1
    else:
        return 0


def genT(X, cutoff=0, positive=True):
    """
    This function creates an array of treatment variables for a sample of observations.

    Parameters
    ----------
    X : arrray[int]
        The sample of observation to compute the treatment values for.

    Returns
    -------
    T: array[int]
        Returns an array with 1's and 0's for each observation depending on wether that observation received treatment.
    """
    T = {}
    for i in range(len(X)):
        T = np.append(T, treatment(X[i],cutoff,positive))
    T = np.delete(T, 0)
    return T


def mu_noack(L, x):
    """
    This function generates the non-random part of outcome function with the DGP used by Noack and Rothe (NR) (2023).

    Parameters
    ----------
    L : int
        The parameter used by NR to define the level of misspecification L={0,10,20,30,40}.
    x : int
        The observation to compute the non-random outcome value for.

    Returns
    -------
    mu: int
        Returns the non-random part of the outcome value for that observation.
    """
    return sign(x) * np.power(x, 2) - L * sign(x) * (
        np.power(x - 0.1 * sign(x), 2) - np.power(0.1, 2) * sign(x)
    ) * indicator(x)


def genY_noack(L, X, epsilon):
    """
    This function generates a sample of outcomes (Y_i's) with the DGP used by Noack and Rothe (NR) (2023).

    Parameters
    ----------
    L : int
        The parameter used by NR to define the level of misspecification L={0,10,20,30,40}.
    X : arrray[int]
        The sample of observation to compute the outcome values for.
    epsilon : arrray[int]
        The vector of random errors.

    Returns
    -------
    Y: arrray[int]
        Returns a vector with the outcome values.
    """
    Y = {}
    for i in range(len(X)):
        Y = np.append(Y, mu_noack(L, X[i]) + epsilon[i])
    Y = np.delete(Y, 0)
    return Y


def mu_basicLinear(tau, alpha, beta, x):
    """
    This function generates the non-random part of the outcome function with a basic linear potential outcomes framework DGP.

    Parameters
    ----------
    tau : int
        The size of the treatment effect.
    alpha: int
        The intercept parameter of the equation.
    beta: int
        The slope parameter of the equation.
    x : int
        The observation to compute the non-random outcome value for.

    Returns
    -------
    mu: int
        Returns the non-random part the outcome value.
    """
    return alpha + beta * x + tau * treatment(x)


def genY_basicLinear(tau, alpha, beta, X, epsilon):
    """
    This function generates a sample of outcomes (Y_i's) with a basic linear potential outcomes framework DGP.

    Parameters
    ----------
    tau : int
        The size of the treatment effect.
    alpha: int
        The intercept parameter of the equation.
    beta: int
        The slope parameter of the equation.
    X : arrray[int]
        The sample of observation to compute the outcome values for.
    epsilon : arrray[int]
        The vector of random errors.

    Returns
    -------
    Y: arrray[int]
        Returns a vector with the outcome values.
    """
    Y = {}
    for i in range(len(X)):
        Y = np.append(Y, mu_basicLinear(tau, alpha, beta, X[i]) + epsilon[i])
    Y = np.delete(Y, 0)
    return Y


# Generation of the Outcomes (Y_i) given the different DGP's


def genY(name, X, tau=0, L=0, alpha=0, beta=0):
    """
    This function generates a sample of observations from the given DGP.

    Parameters
    ----------
    name: string
        The name of the DGP to use.
    X : arrray[int]
        The sample of observation to compute the outcome values for.
    tau : int, Default value: 0
        The size of the treatment effect. For basic and basic linear model.
    L : int, Default value: 0
        The parameter used by NR to define the level of misspecification L={0,10,20,30,40}. For the Noack and Rothe model.
    alpha: int, Default value: 0
        The intercept parameter of the equation. For basic linear model.
    beta: int, Default value: 0
        The slope parameter of the equation. For basic linear model.

    Returns
    -------
    Y: arrray[int]
        Returns a vector with the outcome values.
    """
    epsilon = np.random.normal(0, 0.5, len(X))
    if name == "Noack":
        Y = genY_noack(L, X, epsilon)
    elif name == "Basic Linear":
        Y = genY_basicLinear(tau, alpha, beta, X, epsilon)
    else:
        return NameError("Type of GDP is not recognised")
    return Y


def genOutlier(Y, X, name, nOutliers=1, delta=0.1, cutoff=0):
    """
    This function generates outliers based on different methods

    Parameters
    ----------
    Y : arrray[int]
        The sample of outcomes to generate the outlier(s) value(s) for.
    X : arrray[int]
        The sample of observation to generate the outlier(s) value(s) for.
    name : string
        The name of the method to generate the ouotlier(s).
        Option values: Simple, Simple Outside.
    nOutlier : int
        The number of outliers to generate.
    delta : int
        The size of the donut stripe.

    Returns
    -------
    Y : arrray[int]
        The sample of outcomes with the outlier(s) value(s).
    Outliers : arrray[int]
        An array with 1 if outlier and 0 is not. (Used for coloring the dots on scatter plots)
    """
    Outliers = np.zeros_like(Y)
    # Simple generates outlier(s) inside the donut stripe (right of the cutoff)
    if name == "Simple":
        i = 0
        for j in range(nOutliers):
            notFound = True
            # Find first observation in the stripe and change outcome value to 2.5
            while notFound & (i < len(X)):
                if (X[i] <= delta) & (X[i] > cutoff):
                    Y[i] = 10
                    Outliers[i] = 1
                    notFound = False
                i = i + 1

    # Simple Outised generates outlier(s) just outside the left side of the donut stripe
    if name == "Small Outside Right":
        i = 0
        for j in range(nOutliers):
            notFound = True
            # Find first observation just outside the stripe and change outcome value to 2.5
            while notFound & (i < len(X)):
                if (X[i] <= 2 * delta) & (X[i] > delta):
                    Y[i] = 3.5
                    Outliers[i] = 1
                    notFound = False
                    i = i + 1
                else:
                    i = i + 1

    # Simple Outised generates outlier(s) just outside the right side of the donut stripe
    if name == "Outside Right":
        i = 0
        for j in range(nOutliers):
            notFound = True
            # Find first observation just outside the stripe and change outcome value to 2.5
            while notFound & (i < len(X)):
                if (X[i] <= 2 * delta) & (X[i] > delta):
                    Y[i] = 10
                    Outliers[i] = 1
                    notFound = False
                    i = i + 1
                else:
                    i = i + 1

    # Simple Oposite generates outlier(s) on both sides of the cutoff just outside the donut stripe.
    if name == "Oposite Outside":
        i = 0
        j = 0
        for k in range(nOutliers):
            notFound = True
            # Find first observation just outside the left-side of the stripe and
            # change outcome value to -2.5
            while notFound & (i < len(X)):
                if (X[i] >= -2 * delta) & (X[i] < -delta):
                    Y[i] = -10
                    Outliers[i] = 1
                    notFound = False
                    i = i + 1
                else:
                    i = i + 1

            notFound = True
            # Find first observation just outside the right-side of the stripe and
            # change outcome value to 2.5
            while notFound & (j < len(X)):
                if (X[j] <= 2 * delta) & (X[j] > delta):
                    Y[j] = 10
                    Outliers[j] = 1
                    notFound = False
                    j = j + 1
                else:
                    j = j + 1
        # Simple Oposite generates outlier(s) on both sides of the cutoff just outside the donut stripe.
    if name == "Oposite Inside":
        i = 0
        j = 0
        for k in range(nOutliers):
            notFound = True
            # Find first observation just inside the left-side of the stripe and
            # change outcome value to -2.5
            while notFound & (i < len(X)):
                if (X[i] <= cutoff) & (X[i] > -delta):
                    Y[i] = -10
                    Outliers[i] = 1
                    notFound = False
                    i = i + 1
                else:
                    i = i + 1

            notFound = True
            # Find first observation just inside the right-side of the stripe and
            # change outcome value to 2.5
            while notFound & (j < len(X)):
                if (X[j] >= cutoff) & (X[j] < delta):
                    Y[j] = 10
                    Outliers[j] = 1
                    notFound = False
                    j = j + 1
                else:
                    j = j + 1
   
    if name == "Symetric Inside":
        i = 0
        j = 0
        for k in range(nOutliers):
            notFound = True
            # Find first observation just inside the left-side of the stripe and
            # change outcome value to -10
            while notFound & (i < len(X)):
                if (X[i] <= cutoff) & (X[i] > -delta):
                    Y[i] = -10
                    Outliers[i] = 1
                    notFound = False
                    i = i + 1
                else:
                    i = i + 1

            notFound = True
            # Find first observation just inside the right-side of the stripe and
            # change outcome value to 10
            while notFound & (j < len(X)):
                if (X[j] >= cutoff) & (X[j] < delta):
                    Y[j] = 10
                    Outliers[j] = 1
                    notFound = False
                    j = j + 1
                else:
                    j = j + 1
            notFound = True
            # Find first observation just inside the left-side of the stripe and
            # change outcome value to -10
            while notFound & (i < len(X)):
                if (X[i] <= cutoff) & (X[i] > -delta):
                    Y[i] = 10
                    Outliers[i] = 1
                    notFound = False
                    i = i + 1
                else:
                    i = i + 1

            notFound = True
            # Find first observation just inside the right-side of the stripe and
            # change outcome value to 10
            while notFound & (j < len(X)):
                if (X[j] >= cutoff) & (X[j] < delta):
                    Y[j] = -10
                    Outliers[j] = 1
                    notFound = False
                    j = j + 1
                else:
                    j = j + 1
    return Y, Outliers


# Generation of the Sample X_i's and Y_i's


def genSample(
    name,
    n,
    tau=0,
    alpha=0,
    beta=0,
    L=0,
    cutoff=0,
    outlier=False,
    outlierMethod="",
    nOutliers=1,
    printPlot=False,
):
    """
    Generate a sample for RDD analysis: running variables (X), outcomes (Y), and treatments (T)

    Parameters
    ----------
    name: string, Options: 'Noack', 'Basic', 'Basic Linear'
        The name of the DGP to use to generate the sample.
    n: int
        The size of the sample.
    tau : int, Default value: 0
        The size of the treatment effect. For basic and basic linear model.
    L : int, Default value: 0
        The parameter used by NR to define the level of misspecification L={0,10,20,30,40}. For the Noack and Rothe model.
    alpha: int, Default value: 0
        The intercept parameter of the equation. For basic linear model.
    beta: int, Default value: 0
        The slope parameter of the equation. For basic linear model.
    cutoff : int , Default value: 0
        The treshold of the running variable that determines treatment
    outlier: boolean
        True if sample should have an outlier(s)
    outlierMethod: string
        Name of the outlier generation method to use
    printPlot: boolean, Default value: False
        Defines if a plot is printed with the sample.

    Returns
    -------
    sample: DataFrame
        A dataframe object with the geneated Y (outcomes) and X (running variables) and given T (treatment variables)
    """
    X = np.random.uniform(-1 + cutoff, 1 + cutoff, n)
    Y = genY(name, X, tau, L, alpha, beta)
    Outliers = np.zeros_like(Y)
    if outlier == True:
        Y, Outliers = genOutlier(Y, X, outlierMethod, nOutliers)
    Tr = genT(X)

    # Create sample dataframe
    sample = pd.DataFrame({"Y": Y, "X": X, "Treatment": Tr, "Outlier": Outliers})
    sample.Y = sample.Y.astype(float)
    sample.Treatment = sample.Treatment.astype(float)
    sample.Outlier = sample.Outlier.astype(float)

    # Print plot
    if printPlot == True:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["grey", "red"])
        plt.scatter(X, Y, s=6, c=sample.Outlier, cmap=cmap)
        plt.xlabel("$X_i$")
        plt.ylabel("$Y_i$")

    return sample
