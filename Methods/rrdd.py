import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm



def genExog(sample, intercept=False, jointFit=False):
    ''' 
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
    '''
    # Copy object and delete columns that are not exogenous variables.
    exog = sample.copy() 
    exog = exog.drop('Y',axis='columns')
    if jointFit is False:
        exog = exog.drop('Treatment',axis='columns')
    exog = exog.drop('Outlier',axis='columns')
    # Add an intercept if requested
    if (intercept == True):
        exog = sm.add_constant(exog)
    
    return exog

def fit(name, sample, intercept, cutoff=0, jointFit=False):
    ''' 
    This method will fit a regression based on the different estimation methods on the given sample.

    Parameters
    ----------
    name : string
        The name of the estimation method to use.
        Opiton values : Robust Huber, Robust Tuckey, OLS, Donut.
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
    '''
    # Prepare sample
    exog = genExog(sample,intercept,jointFit)    

    # Estimate regression based on estimation method
    if (name == 'Robust Huber'):
        res = sm.RLM(sample.Y,exog,M=sm.robust.norms.HuberT())
           
    elif (name == 'Robust Tuckey'):
        res = sm.RLM(sample.Y,exog,M=sm.robust.norms.TukeyBiweight())

    elif (name == 'OLS'):
        res = sm.OLS(sample.Y,exog)

    elif (name == 'Donut'):
        sample = sample.loc[np.abs(sample.X-cutoff)>=0.1]
        exog = genExog(sample,intercept,jointFit)
        res = sm.OLS(sample.Y,exog.to_numpy())
        
    else:
        return NameError('Type of Estimation method is not recognised')
    # Fit model and return parameters
    res = res.fit()
    return res
    

def splitFitRD(name,sample,cutoff=0):
    ''' 
    This method estimates the treatment effects based on RDD estimated by 2 regression
    on eeach side of the cutoff.
    
    Parameters
    ----------
    name : string
        The name of the estimation method to use.
        Opiton values : Robust Huber, Robust Tuckey, OLS, Donut.
    sample : DataFrame
        The sample to estimate the regression for.
    cutoff : int
        The value of the threshold in the running variable. 

    Returns
    -------
    tau : int
        The estimated treatment effect.
    '''
    # Split sample at cutoff
    sample_below = sample.loc[sample.X<=cutoff]
    sample_above = sample.loc[sample.X>cutoff]
    params_below = fit(name, sample_below,intercept=True).params
    params_above = fit(name, sample_above,intercept=True).params
    tau =  params_above.iloc[0] - params_below.iloc[0]
    return tau
    
def plotComparison(sample, name1, name2="", name3="",cutoff=0):
    ''' 
    This method plots a figure with the regession lines of the different estimation methods 

    Parameters
    ----------
    sample : DataFrame
        The sample to estimate the regression for.
    name1 : string
        The name of the estimation method to use.
        Opiton values : Robust Huber, Robust Tuckey, OLS, Donut.
    name2 : string ,Default value : ""
        The name of the estimation method to use.
        Opiton values : Robust Huber, Robust Tuckey, OLS, Donut.
    name3 : string ,Default value : ""
        The name of the estimation method to use.
        Opiton values : Robust Huber, Robust Tuckey, OLS, Donut.
    cutoff : int
        The value of the threshold in the running variable.     
    '''
    # Fit regressions and estimate ATE's
    params1_below, params1_above  = splitFit(name1,sample, cutoff, True)
    tau1 =  params1_above.iloc[0] - params1_below.iloc[0]
    if name2 != "":
        params2_below, params2_above = splitFit(name2,sample, cutoff, True)
        tau2 =  params2_above.iloc[0] - params2_below.iloc[0]

    if name3 != "":
        params3_below, params3_above = splitFit(name3,sample, cutoff, True)
        tau3 =  params3_above.iloc[0] - params3_below.iloc[0]

    
    # Plot scatter observations
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["grey","red"])
    plt.figure(figsize=(19,7))
    plt.scatter(sample.X,sample.Y,s=6,c=sample.Outlier,cmap=cmap)
    plt.scatter(sample.X,sample.Y,s=6,c=sample.Outlier,cmap=cmap)
    plt.xlabel('$X_i$')
    plt.ylabel('$Y_i$')

    # Plot regresion lines
    x_below = np.linspace(min(sample.X),cutoff,100)
    x_above = np.linspace(cutoff,max(sample.X),100)
    plt.plot(x_below,params1_below.iloc[0]+params1_below.iloc[1]*x_below, color ='b', linewidth=0.7)
    plt.plot(x_above,params1_above.iloc[0]+params1_above.iloc[1]*x_above, color ='b', linewidth=0.7, label=name1+' (ate: '+str(round(tau1,2))+')')
    if name2 != "":
        plt.plot(x_below,params2_below.iloc[0]+params2_below.iloc[1]*x_below, color ='g', linewidth=0.7)
        plt.plot(x_above,params2_above.iloc[0]+params2_above.iloc[1]*x_above, color ='g', linewidth=0.7, label=name2+' (ate: '+str(round(tau2,2))+')')
    if name3 != "":
        plt.plot(x_below,params3_below.iloc[0]+params3_below.iloc[1]*x_below, color ='purple', linewidth=0.7)
        plt.plot(x_above,params3_above.iloc[0]+params3_above.iloc[1]*x_above, color ='purple', linewidth=0.7, label=name3+' (ate: ' +str(round(tau3,2))+')')

    plt.legend()

def jointFitRD(name,sample,cutoff=0):

    sample['XT'] = sample.X*sample.Treatment
    return fit(name,sample,True,cutoff,True)