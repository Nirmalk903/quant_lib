import pandas as pd
import numpy as np
from quantlib import QuantLib as ql
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

def get_historical_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculate historical volatility using a rolling window.
    
    :param prices: Series of asset prices.
    :param window: Rolling window size in days.
    :return: Series of historical volatility.
    """
    log_returns = np.log(prices / prices.shift(1))
    volatility = log_returns.rolling(window).std() * np.sqrt(252)  # Annualize the volatility
    return volatility

def calculate_option_price(option_type: str, strike: float, spot: float, maturity: datetime,
                        risk_free_rate: float, dividend_yield: float, volatility: float) -> float:
    """
    Calculate the Black-Scholes option price.
    
    :param option_type: 'call' or 'put'
    :param strike: Strike price of the option.
    :param spot: Current spot price of the underlying asset.
    :param maturity: Maturity date of the option.
    :param risk_free_rate: Risk-free interest rate.
    :param dividend_yield: Dividend yield of the underlying asset.
    :param volatility: Volatility of the underlying asset.
    :return: Option price.
    """
    maturity_days = (maturity - datetime.now()).days / 365.0
    d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * maturity_days) / (volatility * np.sqrt(maturity_days))
    d2 = d1 - volatility * np.sqrt(maturity_days)
    
    if option_type == 'call':
        price = (spot * np.exp(-dividend_yield * maturity_days) * norm.cdf(d1) - strike * np.exp(-risk_free_rate * maturity_days) * norm.cdf(d2))
    elif option_type == 'put':
        price = (strike * np.exp(-risk_free_rate * maturity_days) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * maturity_days) * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price

