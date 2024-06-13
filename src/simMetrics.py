import pandas as pd
import statsmodels.api as sm

def compRMSE(pointEstimation, trueValue):
    '''
    This method computes the RMSE of the estimates.

    ''' 
    return [sm.tools.eval_measures.rmse(pointEstimation.OLS,trueValue), sm.tools.eval_measures.rmse(pointEstimation.Huber,trueValue), 
    sm.tools.eval_measures.rmse(pointEstimation.Tukey,trueValue), sm.tools.eval_measures.rmse(pointEstimation.Donut,trueValue)]

# print jarque-bera statistics
    if JB == True:
        print('Jarque-Bera : '+ st.jarque_bera(serie))

def percentV(values):
    return pd.DataFrame({'0.1': values[0].mean(),'0.05': values[1].mean(),'0.01': values[2].mean()})
