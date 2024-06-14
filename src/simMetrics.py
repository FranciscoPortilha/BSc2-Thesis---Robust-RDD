import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.stattools as st


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
