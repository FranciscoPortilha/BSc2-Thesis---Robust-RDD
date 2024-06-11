import numpy as np
import pandas as pd
import statsmodels.api as sm
from methods.sample import genSample
from methods.rrdd import fitRD


def simulation(r,nameSample,n,tau=0,  alpha=0, beta=0, L=0, cutoff=0,b=1, outlier=False, outlierMethod='', nOutliers=0):
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
    t_OLS = {}
    t_RH = {}
    t_RT = {}
    t_D = {}

    # Generate sample and fit models r times
    for i in range(r):
        # Generate sample
        sample = genSample(nameSample,n,tau, alpha, beta,L,cutoff,outlier, outlierMethod, nOutliers, False)
        # Select sample to be used according to bandwidth
        sample = sample.loc[np.abs(sample.X-cutoff)<=b]
        # Estimate models
        t_OLS = np.append(t_OLS, fitRD('OLS',sample,0))
        t_RH = np.append(t_RH, fitRD('Robust Huber',sample,0))
        t_RT = np.append(t_RT, fitRD('Robust Tuckey',sample,0))
        t_D = np.append(t_D, fitRD('Donut',sample,0))
    
    # Adjust the format of the arrays (delete empty first cell)
    t_OLS = np.delete(t_OLS,0)
    t_RH = np.delete(t_RH,0)
    t_RT = np.delete(t_RT,0)
    t_D = np.delete(t_D,0)

    # Create dataframe with simultation results
    simulationResults = pd.DataFrame({'OLS':t_OLS, 'Huber':t_RH, 'Tuckey':t_RT, 'Donut':t_D})
    return simulationResults

def compRMSE(simRes):
    return [sm.tools.eval_measures.rmse(simRes.OLS,2), sm.tools.eval_measures.rmse(simRes.Huber,2), 
    sm.tools.eval_measures.rmse(simRes.Tuckey,2), sm.tools.eval_measures.rmse(simRes.Donut,2)]

def simulations(r,name,n,tau,alpha,beta,cutoff=0,L=0):
    ''' 
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
        The parameter used by NR to define the level of misspecification L={0,10,20,30,40}. For the Noack and Rothe model.
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
    
    '''
    # Run Simulations
    simRes1 = simulation(r,name,n,tau,alpha,beta,cutoff=cutoff,outlier=False)
    simRes2 = simulation(r,name,n,tau,alpha,beta,cutoff=cutoff,outlier=True, outlierMethod='Simple Outside', nOutliers=1)
    simRes3 = simulation(r,name,n,tau,alpha,beta,cutoff=cutoff,outlier=True, outlierMethod='Simple Outside', nOutliers=2)
    simRes4 = simulation(r,name,n,tau,alpha,beta,cutoff=cutoff,outlier=True, outlierMethod='Simple', nOutliers=1)
    simRes5 = simulation(r,name,n,tau,alpha,beta,cutoff=cutoff,outlier=True, outlierMethod='Simple', nOutliers=2)
    simRes6 = simulation(r,'Noack',n,tau,L=40,cutoff=cutoff,b=0.5,outlier=False)

    # Compute Root Mean Squared Error
    rmse1 = compRMSE(simRes1)
    rmse2 = compRMSE(simRes2)
    rmse3 = compRMSE(simRes3)
    rmse4 = compRMSE(simRes4)
    rmse5 = compRMSE(simRes5)
    rmse6 = compRMSE(simRes6)

    # Create dataframe with results
    result_1 = pd.DataFrame({'Mean':simRes1.mean(),'St. Dev.': simRes1.std(), 'RMSE':rmse1})
    result_2 = pd.DataFrame({'Mean':simRes2.mean(),'St. Dev.': simRes2.std(), 'RMSE':rmse2})
    result_3 = pd.DataFrame({'Mean':simRes3.mean(),'St. Dev.': simRes3.std(), 'RMSE':rmse3})
    result_4 = pd.DataFrame({'Mean':simRes4.mean(),'St. Dev.': simRes4.std(), 'RMSE':rmse4})
    result_5 = pd.DataFrame({'Mean':simRes5.mean(),'St. Dev.': simRes5.std(), 'RMSE':rmse5})
    result_6 = pd.DataFrame({'Mean':simRes6.mean(),'St. Dev.': simRes6.std(), 'RMSE':rmse6})
    
    # Create one dataframe with all results
    multiCol1 = pd.MultiIndex.from_arrays([['Scenario 1', 'Scenario 1', 'Scenario 1',
                                           'Scenario 2', 'Scenario 2', 'Scenario 2',
                                           'Scenario 3', 'Scenario 3', 'Scenario 3'], 
                                           ['Mean','St.Var','RMSE',
                                            'Mean','St.Var','RMSE',
                                            'Mean','St.Var','RMSE',]])
    multiCol2 = pd.MultiIndex.from_arrays([['Scenario 4', 'Scenario 4', 'Scenario 4',
                                           'Scenario 5', 'Scenario 5', 'Scenario 5',
                                           'Scenario 6', 'Scenario 6', 'Scenario 6'], 
                                            ['Mean','St.Var','RMSE',
                                            'Mean','St.Var','RMSE',
                                            'Mean','St.Var','RMSE']])
    row = ['OLS', 'Huber', 'Tuckey', 'Donut']
    Results1 = pd.DataFrame(np.transpose(np.array([simRes1.mean(),simRes1.std(),rmse1, 
                                                  simRes2.mean(),simRes2.std(),rmse2, 
                                                  simRes3.mean(),simRes3.std(),rmse3])),
                                                    columns=multiCol1, index=row)
    
    Results2 = pd.DataFrame(np.transpose(np.array([simRes4.mean(),simRes4.std(),rmse4, 
                                                  simRes5.mean(),simRes5.std(),rmse5, 
                                                  simRes6.mean(),simRes6.std(),rmse6])),
                                                    columns=multiCol2, index=row)
    return Results1 , Results2