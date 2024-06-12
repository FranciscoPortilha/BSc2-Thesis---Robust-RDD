import statsmodels.api as sm

def compRMSE(pointEstimation, trueValue):
    '''
    This method computes the RMSE of the estimates.

    ''' 
    return [sm.tools.eval_measures.rmse(pointEstimation.OLS,trueValue), sm.tools.eval_measures.rmse(pointEstimation.Huber,trueValue), 
    sm.tools.eval_measures.rmse(pointEstimation.Tuckey,trueValue), sm.tools.eval_measures.rmse(pointEstimation.Donut,trueValue)]

# print jarque-bera statistics
    if JB == True:
        print('Jarque-Bera : '+ st.jarque_bera(serie))