from collections import OrderedDict
import datetime
import pandas as pd
import numpy as np
from data_loader import PortLoader
import os
import math
from utils import *

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
    def evaluate(self):
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
                port_alpha, port_beta = self.jensen_alpha(port, mkt_port)
                
                temp['Params'].append(self.port_params['port_name'])
                temp['Beta'].append(port_beta)
                temp['Sharpe'].append(port_sharpe*math.sqrt(252))
                temp['Alpha'].append(port_alpha)

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

    def jensen_alpha(self, port, benchmark, rf=0.22/252):
        """
        Returns excess return of a portfolio according to
        Rp - (rf + beta x (rm - rf))
        """
        port_ret = port['Daily Return']
        mkt_ret  = benchmark['Daily Return']
        port_beta, port_alpha = np.polyfit(benchmark['Daily Return'], port['Daily Return'], 1)

        port_ret_mean = port_ret.mean()
        mkt_ret_mean = mkt_ret.mean()

        port_alpha = port_ret_mean - (rf + port_beta * (mkt_ret_mean - rf))
        port['Excess Return'] = port['Daily Return'] - benchmark['Daily Return']
        return port_alpha, port_beta
        
    def report(self):
        print('')
        
        return