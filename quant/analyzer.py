from collections import OrderedDict
import datetime
import pandas as pd
import numpy as np
from data_loader import PortLoader
import os
import math
from utils import *
from scipy.stats import (
    norm as _norm, linregress as _linregress
)


class Analyzer():
    """A class that abstracts the analyzing process of portfolio with corresponding training weights

    Attributes:
        data_path: 
            relative path to the data folder, e.g. './data'
    """
    def __init__(self, data_path):
        self.data_path = data_path
    
    def evaluate():
        return NotImplementedError


class ReturnAnalyzer(Analyzer):
    """A class that abstracts the loading process of portfolio with corresponding training weights

    Attributes:
        data_path: 
            relative path to the data folder, e.g. './data'
    """
    def __init__(self, data_path):
        super().__init__(data_path)
        
        port_data = PortLoader(data_path)
        self.port_params = port_data.port_params
        self.mkt_port = port_data.mkt_port

    @timer
    def evaluate(self, alt=0):
        """
        Returns the benchmark of every portfolio.
        """
        print("Begin Evaluating Returns...")
        temp = {'Params': [],
                'Beta': [],
                'Sharpe':[],
                'Alpha':[]
                }
        for i in range(len(self.data_path)):
            filepath = self.data_path[i]
            filename = os.path.basename(filepath)
            if 'param' in filename and 'long' in filename:
                port = pd.read_csv(filepath)
                port = port.dropna()
                # port['Daily Return'] = port['Daily Return'].apply(lambda x: x/scale)

                # Benchmark
                start, end = port['Dates'].iloc[0], port['Dates'].iloc[-1]
                mkt_port = self.mkt_port[self.mkt_port.index.to_series().between(start, end)]
                mkt_port = mkt_port.dropna()

                # Report Metrics
                port_sharpe = self.sharpe_ratio(port)
                port_alpha, port_beta = self.jensen_alpha(port, mkt_port, alt)
                information_ratio = self.information_ratio(port, mkt_port)
                var = self.value_at_risk(port)
                r_2 = self.r_squared(port, mkt_port)
                tail_ratio = self.tail_ratio(port, mkt_port)
                
                temp['Params'].append(self.port_params['port_name'])
                temp['Beta'].append(port_beta)
                temp['Sharpe'].append(port_sharpe*math.sqrt(252))
                temp['Alpha'].append(port_alpha)
                temp['Information Ratio'].append(information_ratio)
                temp['Value at Risk'].append(var)
                temp['R_Squared'].append(r_2)
                temp['Tail Ratio'].append(tail_ratio)
                
        df_metric = pd.DataFrame(data=temp)
        return df_metric
    

    def sharpe_ratio(self, port, rf=0.22/252):
        """
        This function returns the daily sharpe ratio of a portfolio
        as (E[R] - rf)/Std(R)
        """
        port_ret = pd.DataFrame(port['Daily Return'].dropna())
        mean = port_ret.mean()
        std = port_ret.std()
        sharpe = (mean - rf)/std
        return float(sharpe)

    def jensen_alpha(self, port, benchmark, rf=0.22/252, alt=0):
        """
        Returns excess return of a portfolio according to
        Rp - (rf + beta x (rm - rf))
        """
        port_ret = port['Daily Return']
        mkt_ret  = benchmark['Daily Return']
        if alt == 0:
            port_beta, port_alpha = np.polyfit(benchmark['Daily Return'], port['Daily Return'], 1)
        else:
            cov_matrix = np.cov(port_ret, mkt_ret)
            port_beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        port_ret_mean = port_ret.mean()
        mkt_ret_mean = mkt_ret.mean()

        # 
        port_alpha = port_ret_mean - (rf + port_beta * (mkt_ret_mean - rf))
        port['Excess Return'] = port['Daily Return'] - benchmark['Daily Return']
        return port_alpha, port_beta

    def information_ratio(self, port, benchmark, rf=0.2/252):
        """
        Calculates the information ratio.
        """
        excess_ret = port['Daily Retybr'] - benchmark['Daily Return']
        
        return excess_ret.mean() / excess_ret.std()

    def r_squared(self, port, benchmark):
        """
        Calculates the linear fit of the portfolio to the market
        """
        r_val = _linregress(port['Daily Return'], benchmark['Daily Return'])
        return r_val**2
    
    def tail_ratio(self, port, cutoff=0.95):
        """
        Calculates the two-end tail ratio.
        """
        return abs(port['Daily Return'].quantile(cutoff) / port['Daily Return'].quantile(1-cutoff))

    def value_at_risk(port, sigma=1, confidence=0.95):
        """
        Calculates the daily value-at-risk"""
        port_ret = port['Daily Return']
        mu, sigma = port_ret.mean(), port_ret.std()

        return _norm.ppf(1-confidence, mu, sigma)

    def rolling_greeks(self, port, benchmark, periods=252, rf=0.2/252):
        """
        Calculates rolling alpha and beta of a portfolio
        """
        df_comp = pd.DataFrame(data={
            'port':port['Daily Return']
            'benchmark': benchmark['Daily Return']})
        df_comp.fillna(0)
        corr = df_comp.rolling(int(periods)).corr().unstack()['port']['benchmark']
        std = df_comp.rolling(int(periods)).std()

        beta = corr * std['port'] / std['benchmark']
        alpha = df_comp['port'].mean() - beta * df_comp['benchmark'].mean()

        return pd.DataFrame(index=port.index, data={
            "beta": beta,
            "alpha": alpha
        })


