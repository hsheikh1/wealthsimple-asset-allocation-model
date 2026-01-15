"""
risk_analytics.py

A quantitative finance library for portfolio construction and risk management.
Includes modules for:
- Robust Risk Estimation (VaR, CVaR, Drawdowns)
- Portfolio Optimization (MSR, GMV, Risk Parity)
- Asset Allocation Modeling (Black-Litterman, CPPI)

Original framework adapted for institutional asset allocation analysis.
Curated by: Haroon Sheikh, CFA
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, jarque_bera
from scipy.optimize import minimize
from typing import Union, Tuple, List

# ==========================================
# 1. CORE RISK METRICS
# ==========================================

def annualize_rets(r: pd.Series, periods_per_year: int = 12) -> float:
    """
    Computes the annualized return of a series of returns.
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1


def annualize_vol(r: pd.Series, periods_per_year: int = 12) -> float:
    """
    Computes the annualized volatility of a series of returns.
    """
    return r.std() * (periods_per_year**0.5)


def sharpe_ratio(r: pd.Series, riskfree_rate: float, periods_per_year: int = 12) -> float:
    """
    Computes the annualized Sharpe Ratio of a set of returns.
    """
    # Convert the annual riskfree rate to per period
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def drawdown(return_series: pd.Series) -> pd.DataFrame:
    """
    Takes a time series of asset returns.
    Returns a DataFrame with columns for:
    - Wealth index
    - Previous peaks
    - Percentage drawdown
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index, 
        "Previous Peak": previous_peaks, 
        "Drawdown": drawdowns
    })


def skewness(r: pd.Series) -> float:
    """
    Computes the skewness of the supplied Series or DataFrame.
    Alternative to scipy.stats.skew().
    """
    demeaned_r = r - r.mean()
    # Use the population standard deviation, so set ddof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / sigma_r**3


def kurtosis(r: pd.Series) -> float:
    """
    Computes the kurtosis of the supplied Series or DataFrame.
    Alternative to scipy.stats.kurtosis().
    """
    demeaned_r = r - r.mean()
    # Use the population standard deviation, so set ddof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / sigma_r**4


def is_normal(r: pd.Series, level: float = 0.01) -> bool:
    """
    Applies the Jarque-Bera test to determine if a Series is normal.
    Returns True if the hypothesis of normality is accepted, False otherwise.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = jarque_bera(r)
        return p_value > level

# ==========================================
# 2. VALUE AT RISK (VaR) MODELS
# ==========================================

def var_historic(r: pd.Series, level: int = 5) -> float:
    """
    Returns the historic Value at Risk at a specified level.
    i.e. returns the number such that "level" percent of the returns
    fall below that number.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_gaussian(r: pd.Series, level: int = 5, modified: bool = False) -> float:
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame.
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification to adjust for non-normality.
    """
    # Compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # Modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r: pd.Series, level: int = 5) -> float:
    """
    Computes the Conditional VaR (Expected Shortfall) of Series or DataFrame.
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

# ==========================================
# 3. PORTFOLIO OPTIMIZATION
# ==========================================

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights.
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the volatility of a portfolio from a covariance matrix and constituent weights.
    """
    vol = (weights.T @ covmat @ weights)**0.5
    return vol 


def minimize_vol(target_return: float, er: pd.Series, cov: pd.DataFrame):
    """
    Returns the optimal weights that achieve the target return
    while minimizing volatility.
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    # Construct the constraints
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    
    results = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1, return_is_target),
                       bounds=bounds)
    return results.x


def msr(riskfree_rate: float, er: pd.Series, cov: pd.DataFrame):
    """
    Returns the weights of the portfolio that gives the Maximum Sharpe Ratio.
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    # Construct the constraints
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the Sharpe ratio (since we minimize).
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    results = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return results.x


def gmv(cov: pd.DataFrame):
    """
    Returns the weights of the Global Minimum Volatility portfolio.
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

# ==========================================
# 4. ROBUST ESTIMATORS & RISK PARITY
# ==========================================

def sample_cov(r: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Returns the sample covariance of the supplied returns.
    """
    return r.cov()


def cc_cov(r: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Estimates a covariance matrix using the Constant Correlation model.
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # Calculate mean correlation
    rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)


def shrinkage_cov(r: pd.DataFrame, delta: float = 0.5, **kwargs) -> pd.DataFrame:
    """
    Covariance estimator that shrinks between the Sample Covariance and 
    the Constant Correlation Estimators.
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta * prior + (1 - delta) * sample


def risk_contribution(w, cov):
    """
    Compute the contributions to risk of the constituents of a portfolio.
    """
    total_portfolio_var = portfolio_vol(w, cov)**2
    # Marginal contribution of each constituent
    marginal_contrib = cov @ w
    risk_contrib = np.multiply(marginal_contrib, w.T) / total_portfolio_var
    return risk_contrib


def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that ensures the contributions 
    to portfolio risk match the target_risk.
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions.
        """
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs - target_risk)**2).sum()
    
    results = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return results.x


def equal_risk_contributions(cov: pd.DataFrame):
    """
    Returns the weights of the Equal Risk Contribution (Risk Parity) portfolio.
    """
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1/n, n), cov=cov)


def weight_erc(r: pd.DataFrame, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the ERC portfolio given a covariance matrix estimator.
    """
    est_cov = cov_estimator(r, **kwargs)
    return equal_risk_contributions(est_cov)


def summary_stats(r: pd.DataFrame, riskfree_rate: float = 0.03) -> pd.DataFrame:
    """
    Return a DataFrame that contains aggregated summary stats for the returns.
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })