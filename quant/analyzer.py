from collections import OrderedDict
import datetime
import pandas as pd
import numpy as np
from data_loader import PortLoader
import os
from utils import *

class Analyzer():
    """A class that abstracts the analyzing process of portfolio with corresponding training weights

    Attributes:
        data_path: 
            relative path to the data folder, e.g. './data'
    """
    def ___init__(self, data_path):
        self.__init__()
        self.data_path = data_path
    
    def evaluate():
        return NotImplementedError


class ReturnAnalyzer(Analyzer):
    """A class that abstracts the loading process of portfolio with corresponding training weights

    Attributes:
        data_path: 
            relative path to the data folder, e.g. './data'
    """
    def ___init__(self, data_path):
        super().___init__(data_path)
        self.port_params = PortLoader(data_path).port_params
    
    @timer
    def evaluate(self):
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
                mkt_port = mkt_port[mkt_port.index.to_series().between(start, end)]
                mkt_port = mkt_port.dropna()

                # Report Metrics
                port_sharpe = self.sharpe_ratio(port)
                port_alpha, port_beta = self.jensen_alpha(port, mkt_port)
                
                temp['Params'].append(self.port_params['port name'])
                temp['Beta'].append(port_beta)
                temp['Sharpe'].append(port_sharpe)
                temp['Alpha'].append(port_alpha)

        df_metric = pd.DataFrame(data=temp)
        return df_metric

    def sharpe_ratio(self, rf=0):
        """
        This function returns the daily sharpe ratio of a portfolio
        as (E[R] - rf)/Std(R)
        """
        port_ret = pd.DataFrame(self.port['Daily Return'].dropna())
        mean = port_ret.mean()
        std = port_ret.std()
        sharpe = (mean - rf)/std
        return float(sharpe)

    def jensen_alpha(self, benchmark, rf=0):
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