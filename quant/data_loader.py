import numpy as np
import pandas as pd
import os
from utils import *
import yfinance as yf

class PortLoader():
    """
    This class takes in various data source to construct a portfolio in the same format.
    Attributes:
        path_list: 
            relative path to the data folder, e.g. './data'
        port_params:
            a list of portfolio with respective parameters
    """
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.port_params = self.load_port()
        self.mkt_port = self.load_mkt(mkt_ticker='XU100.IS')

    def load_mkt(self, mkt_ticker: str):
        """
        Load the benchmark portfolio for analysis
        """
        print("Download Market Benchmark...")
        mkt_port = yf.download(mkt_ticker, start='2022-01-01')
        mkt_port['Daily Return'] = (mkt_port['Close'] / mkt_port['Close'].shift(1)) -1
        return mkt_port.dropna()

    @timer
    def load_port(self):
        print("Loading Portfolio Data...")
        temp = {'port_name': [],
                'long_short': [],
                'holding_period': [],
                'training_weight': [],
                'proportion of short': [],
                'scale of returns_long': [],
                'scale of returns_short': []
                }
        for filepath in self.data_path:
            filename = os.path.basename(filepath)
            if 'param' in filename and 'results' not in filename:
                temp['port_name'].append(filename)
                params = filename.split('_')
                temp['long_short'].append(int('longs' in params[0]))
                temp['holding_period'].append(float(params[1]))
                temp['training_weight'].append(params[6])
                temp['proportion of short'].append(float(params[8])/10)
                temp['scale of returns_long'].append(float(params[10])/100)
                temp['scale of returns_short'].append(float(params[12])/100)
                # temp['lookback period_long'].append(float())
                # TO-DO: add remaining parameters to the portfolio
            
        return pd.DataFrame(data=temp)



