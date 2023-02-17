#####
# general modules
import pandas as pd
import os
import time
import datetime as dt
from IPython.display import clear_output
from tqdm.notebook import tqdm
import copy
import multiprocessing
import random
import datatable
#####
# data modules
import pandas_datareader.data as reader
import matplotlib.pyplot as plt
import numpy as np
import math
import datatable as dt2
#####
# annealing modules
import dwave as d
import dimod
import neal
from pandas.tseries.offsets import BDay
import ast
#####

def get_key(filename):
    sp = filename.split("/")[-1].split("_")
    ret = "(" + str(sp[6]) + ", " + str(sp[8]) + ", " + str(sp[10]) + ", " + str(sp[15]) + ", " + str(sp[21]) + ")"
    if sp[0]=="shorts":
        ret = "(" + str(sp[6]) + ", " + str(sp[8]) + ", " + str(sp[12]) + ", " + str(sp[18]) + ", " + str(sp[24].split(".")[0]) + ")"
    return ret

def rem_vol(param_data, tolerance=0.2, fwd=1):
    """
    rem_vol removes extreme fluctuations in the prices of assets. 

    :param param_data: the original dataset.
    
    :param tolerance: specifies the maximum fluctuation between any two price points that should be allowed, and is 
    20% by default. 

    :param fwd: specifies the number of prices that should be averaged to determine the the prices which should be removed.
    
    :return: returns the original dataset as a pandas dataframe with extreme price fluctuations removed.
    """
    data = param_data.copy(deep=True)
    rows, columns = data.shape

    for c in range(columns):
        for r in range(1, rows):
            
            if (not np.isnan(data.iloc[r,c])) and (not np.isnan(data.iloc[r-1,c])):
                if abs((data.iloc[r,c] - data.iloc[r-1,c])/data.iloc[r-1,c]) > tolerance:
                    
                    if r != rows-1:
                        i = r+1
                        count = 0
                        avgP = 0

                        # compute average of future non-nan values until dataset ends or the specified limit is reached
                        while i < rows and count < fwd:
                            if not np.isnan(data.iloc[i,c]):
                                avgP+=data.iloc[i,c]
                                count+=1
                            i+=1

                        avgP /= count

                        if abs(data.iloc[r,c]-avgP) < abs(data.iloc[r-1,c]-avgP):
                            data.iat[r-1, c] = np.nan
                            print(data.columns[c], "changed at", data.index[r-1])
                        else:
                            data.iat[r,c] = np.nan
                            print(data.columns[c], "changed at", data.index[r])

                    else:
                        data.iat[r,c] = np.nan

    return data

def drop_vol(data, tolerance=0.8):
    """
    drop_vol removes assets that experience significant price increases in short periods of time
    
    :param data: the original dataset in the form of a pandas dataframe
    
    :param tolerance: the maximum percentage price increase that should be tolerated between two time periods, set to 0.8 (80%)
    by default
    
    :return: returns a pandas data frame containing the original values with assets experiencing significant price fluctuations
    dropped
    """
    dropped = []
    for c in data.iteritems():
        col_dropped = False
        prev = c[1].iloc[0]
        for r in c[1].iloc[1:].iteritems():
            if not col_dropped:
                if abs((r[1] - prev)/prev) >  tolerance:
                    dropped.append(c[0])
                    col_dropped = True
                    print("Dropped " + str(c[0]) + " on " + str(r[0]) + " because of a significant price fluctutation")
            prev = r[1]
    return data.drop(dropped, axis=1)

def remove_assets(data, tolerance=0.1):
    """
    remove_assets removes assets with a proportion of nan values greater than the specified tolerance.
    
    :param data: the original dataset.
    
    :param tolerance: specifies the maximum proportion of nan values that should be accepted for an asset class, is set to
    0.1 (10%) by default.
    
    :return: returns the given data set as a pandas dataframe with the asset classes that have a proportion of nan values 
    greater than the specified tolerance dropped
    """
    col_names = data.columns.values
    rows, columns = data.shape
    dropped = []
    for col_name in col_names:
        if (data[col_name].isna().sum()/rows > tolerance):
            dropped.append(col_name)
            print("Dropped", col_name, "since it has NaNs.")
    return data.drop(dropped, axis=1)

def rep_nan(param_data, strat="mean"):
    """
    rep_nan replaces nan values in data set.
    
    :param param_data: the original dataset.

    :param strat: specifies whether the outlying variable should be replaced with a mean of neighboring values, or by a 
    value that is randomly sampled from a normal distribution, and is "mean" by default. To select the normal distribution 
    strategy use the parameter "norm" instead.
    
    :return: returns a pandas dataframe that contains the values of the original dataset, with the nan values replaced according
    to the selected strategy.
    """
    
    data = param_data.copy(deep=True)
    
    if strat == "mean" or strat == "norm":
        rows, columns = data.shape
        for c in range(columns):
            if strat == "norm":
                st_dev = data.iloc[:, c].std()
            for r in range(rows):
                if np.isnan(data.iloc[r,c]):

                    prev, nxt = -1, -1
                    i, k = r, r


                    while i > -1:
                        if not np.isnan(data.iloc[i, c]):
                            prev = data.iloc[i, c]
                            break
                        i-=1


                    while k < rows:
                        if not np.isnan(data.iloc[k, c]):
                            nxt = data.iloc[k, c]
                            break
                        k+=1
                        
                    if (prev>=0) or (nxt>=0):
                        
                        if (prev>=0) and (nxt>=0):
                            neighbor_mean = (prev+nxt)/2
                            
                        if prev>=0 and nxt<0:
                            neighbor_mean = prev
                            
                        elif prev<0 and nxt>=0:
                            neighbor_mean = nxt
                        
                        if strat == "mean":
                            # replace nan value with the mean of its two nearest neighbors
                            data.iat[r,c] = neighbor_mean
                        elif strat == "norm":
                            # replace nan value from a random normal distribution with the mean as the average of its two
                            # neighbors, and the standard deviation as the standard deviation of the prices of the given asset
                            data.iat[r,c] = np.random.normal(neighbor_mean, st_dev)
                            
                        # print out replaced values
                        print("Replaced " + str(data.columns[c]) + " price on date " + str(data.index[r]))

                    else:
                        data.iat[r,c] = 0
    else:
        print("Select a valid strategy")
        
    return data

def organize_results(results, col_names=dict(), index_name=""):
    """
    organize_results outputs backtest results as a pandas data frame with labels

    :param results: this is expected to be the output from running backtests with a set of given parameters, where the form
    of results is expected to be an array of 8-D tuples
    
    :param col_names: this is expected to be a dictionary containing the column names for the given backtest result. The keys
    should be integers in the range [0, len(results)-1], and the values should be the column names associated with each column
    
    :param index_name: this is expected to be a string which is the name of the column that should be used as the index of the
    pandas dataframe. If a column name is not provided, the name of the first column, or the column with index 0, will be used.

    :return: a pandas data frame containing an organized and labelled version of the given backtest results.
    """
    # organizing our results
    if not len(col_names):
        col_names = {0:"Dates", 1:"Stocks", 2:"Number of Stocks", 3:"Energy", 4:"Execution Time", 5:"Portfolio Value", 6:"Daily Return", 7:"Cumulative Return", 8:"Daily Long Return", 9:"Cumulative Long Return", 10:"Daily Short Return", 11:"Cumulative Short Return"}
    
    if len(col_names)!=len(results[0]):
        print("WARNING: LENGTH OF COLUMN NAMES DOES NOT MATCH NUMBER OF COLUMNS IN BACKTEST RESULTS")
    
    if not len(index_name):
        try:
            index_name = col_names[0]
        except:
            print("ERROR: YOUR COLUMN NAMES DO NOT HAVE A COLUMN NAME WITH INDEX ZERO")
            print("picking a random column to be index...")
            index_name = col_names[list(col_names.keys())[random.randrange(len(col_names))]]
            print("picked " + index_name)
       
    final_output = pd.DataFrame(results)
    final_output = final_output.rename(columns=col_names)
    
    try:
        final_output = final_output.set_index(index_name)
    except:
        print("ERROR: PROVIDE A VALID INDEX NAME")
        return None
        
    return final_output

def print_results(results, benchmark, output=False, res_color='g', bench_color='r'):
    """
    print_results charts the total value of the of the stock portfolio for the given backtest and prints out the cumulative return
    achieved by the portfolio for the given time period. This function has no return value.

    :param results: this is expected to be the output from running backtests with a set of given parameters, where the form
    of results is expected to be an array of 8-D tuples
    
    :param benchmark: this is expected to be a pandas series containing the historical price information of the benchmark to
    compare against the model's results.

    :param output: specifies the type of input to be printed. If set to false, the function will expect an  array of 8-D tuples,
    otherwise, the array will expect a pandas dataframe. This parameter is set to False by default
    
    :param res_color: this is an optional parameter that specifies the color in which the model's results  will be plotted,
    and is set to green by default.
    
    :param bench_color: this is an optional parameter that specifies the color in which the benchmark will be plotted,
    and is set to red by default.
    """
    if not output:
        init_v = y[0][5]
        times = [x[0] for x in results]
        vals = [(y[5]-init_v)/init_v for y in results]
        start = str(results[0][0])
        end = str(results[len(results)-1][0])
        return_rate = str(100*results[len(results)-1][7])
    else:
        init_v = results['Portfolio Value'][0]
        times = [x for x in results['Dates']]
        vals = [(y-init_v)/init_v for y in results['Portfolio Value']]
        start = str(times[0])
        end = str(times[len(results)-1])

    return_rate = 100*results.iloc[len(results)-1]['Cumulative Return']
    
    init_b = benchmark[0]
    times_b = [xb for xb in benchmark.index]
    vals_b = [(yb-init_b)/init_b for yb in benchmark]
    
    return_rate = 100*vals[len(vals)-1]
    return_rate_b = 100*vals_b[len(vals_b)-1]
    
    m_b_ratio = 100*((return_rate-return_rate_b)/return_rate_b)
    
    
    plt.plot(times_b, vals_b, bench_color)
    plt.plot(times, vals, res_color)
    
    plt.show()
    print("The model had a return rate of "  + str(return_rate)  + "% in the time period " + start + "-" + end)
    print(" ")
    print("The benchmark had a return rate of "  + str(return_rate_b)  + "% in the time period " + start + "-" + end)
    print(" ")
    print("The model performed " + str(m_b_ratio) + "% better in the time period " + start + "-" + end + " compared to the benchmark.")

# WARNING: don't open csv file while it is being updated, it will halt the backtest
def update_progress(output_name, execution_day, info_vals):
    """
    update_progress will update a csv file containing the results of the model while the model is still being run to save progress made.

    :param output_name: the name of the csv file which will be updated

    :param execution_day: the date of the row being added to the file

    :param stock_names: the stocks in the portolio on execution day

    :param number_of_stocks: the number of stocks in the portfolio on execution day

    :param lowest_energy: the energy calculated by the model for the solution to the optmization problem on execution day

    :param execution_time: the amount of time spent in seconds the model took in computing the solution to the optimization problem

    :param cap: the total value of the portfolio on execution day

    :param daily_return: the daily return generated by the portfolio on execution day

    :param cum_return: the cumulative return generated by the portfolio on execution day
    """
    
    if os.path.exists(output_name):
        # read file
        #historical_port_dt  = datatable.fread(output_name)
        #historical_portfolio = historical_port_dt.to_pandas()
        historical_portfolio = pd.read_csv(output_name)
        
        historical_portfolio = historical_portfolio.set_index('Dates')
        
        # update file with given values
        historical_portfolio.loc[execution_day] = info_vals
    
    else:
        # initializing a new otput file with column names
        historical_portfolio = organize_results([(execution_day, *info_vals)])
    # output file
    historical_portfolio.to_csv(output_name)

# should not be used any longer other than for purposes of experimenting with upper triangular matrices multiplied by a desired
# coefficient. For all other cases, numpy offers a better alternative
def matr_to_upper(matr, mult=1):
    """
    matr_to_upper converts a given matrix into upper triangular form.
    
    :param matr: a matrix, provided as a pandas dataframe.
    
    :param mult: the coefficient by which the upper half of the matrix is multiplied, set to 1 by default.
    
    :return: a matrix in upper triangular form, provided as a pandas dataframe.
    
    """
    m = matr.copy(deep=True)
    rows, columns = m.shape
    
    for c1 in range(columns):
        for r1 in range(c1+1, rows):
            m.iat[r1,c1]=0
    
    for r2 in range(rows):
        for c2 in range(r2+1, columns):
            m.iat[r2,c2]*=mult
    
    return m

def calculate_returns(param_data, normalize=True):
    """
    returns computes the return generated by each asset class during the period of the dataset provided
    
    :param param_data: the original dataset in current window.

    :param normalize: this parameter specifies whether the returns should be normalized. If set to True,
    it will divide the returns generated by the number of days present in the given dataset.
    This parameter is set to True by default.
    
    :return: returns an array containing the returns generated by each dataset
    """
    data = param_data.copy(deep=True)
    ret = {}
    
    factor = 1
    if normalize:
        factor = len(data)
        
    for c in data.iteritems():
        ret[c[0]] = (c[1][-1]-c[1][0])/(c[1][0]*factor)
        
    del data

    return ret

def reformulate_upper_triangular(matr, drop=False):
    """
    reformulate_upper_triangular reformulates upper triangular covariance matrices into a python dictionary where all key-value 
    pairs are of the form (r, c):v, where r is the row label, c is the column label, and v is the covariance between the assets
    represented by r, and c. (r, c) values are unique, meaning that if  (r,c) is in the dict, (c,r) will not be in the dict.
    Furthermore, for any label l, (l, l) will be in the dict.
    
    :param matr: represents the original covariance matrix in upper triangular form which will be reformulated
    
    :param drop: specifies whether the diagonal should be dropped, is set to False by default
    
    :return: returns a python dictionary fitting the aforementioned specifications.
    
    """
    m = matr.copy(deep=True)
    reformed = dict({})
    rows, columns = m.shape
    
    if drop:
        offset = 1
    else:
        offset = 0
    
    for r in range(rows):
        for c in range(r + offset, columns):
            reformed[(m.index[r], m.columns[c])] = m.iloc[r,c]
    return reformed

# This method was created for the initial version of our model and is no longer in use
def combine_return_variances(variance_vals, return_vals, var_coeff=1, ret_coeff=1):
    """
    combine_return_variances computes coefficients which combine the values of return rates and variances of stock prices
    
    :param variance_vals: a dict where the keys are stock labels and the values are the variance of stock price series
    
    :param return_vals: a dict where the keys are stock labels and the values are the return rate of stock price series 
    in the given time period
    
    :param var_coeff: coefficient of the stock price series variance
    
    :param ret_coeff: coefficient of the stock price series return rate
    
    :return: returns a dictionary containing coefficients combining the variance and return rate of stock price series
    """
    coefficients = dict({})
    
    for k in variance_vals.keys():
        coefficients[k] = var_coeff*variance_vals[k] - ret_coeff*return_vals[k]
        
    return coefficients

def construct_quadratic_term_bin(upper_cov_mat_l, upper_cov_mat_s, w_l, w_s, shorts, longs, cs2, cs3, c2, c3, es2, es3, e2, e3, w_shorts):
    """
    construct_quadratic_term_bin constructs the quadratic term for the binary weighting scheme
    
    :param upper_cov_mat_l: a covariance matrix for stock price data relating to long positions provided in the form of an upper triangular pandas dataframe
    
    :param upper_cov_mat_s: a covariance matrix for stock price data relating to short positions provided in the form of an upper triangular pandas dataframe
    
    :param w: the number of bits in budget expansion
    
    :param longs: a boolean parameter which determines whether or not long positions should be taken
    
    :param shorts: the factor by which the returns for short positions will be scaled. If this value is equal to zero, it
    will be assumed that the portfolio only contains long positions
    
    :param longs: a boolean variable to determine whether long positions should be taken
    
    :param cs2: the factor by which the variance terms of short positions will be multiplied
    
    :param cs3: the factor by which the covariance terms of short positions will be multiplied
    
    :param c2: the factor by which the variance terms of long positions will be multiplied
    
    :param c3: the factor by which the covariance terms of long positions will be multiplied
    
    :param es2: the factor by which the variance terms of short positions will be exponentiated
    
    :param es3: the factor by which the covariance terms of long positions will be exponentiated
    
    :param e2: the factor by which the variance terms of short positions will be exponentiated
    
    :params e3: the factor by which the covariance terms of long positions will be exponentiated
    
    :w_shorts: a boolean variable to determine whether shorts should be assigned different weights
    
    :return: a pandas dataframe containing the quadratic term for the binary weighting scheme
    """
    m_l = upper_cov_mat_l.copy(deep=True)
    m_s = upper_cov_mat_s.copy(deep=True)
    
    end_row_l=0
    end_row_s=0
    # iterating through each covariance term, including individual stock variances
    quad = dict()
    l_quad = dict()
    s_quad = dict()
    
    # this constructs the quadratic term for long positions if long positions are to be included in the portfolio
    if longs:
        for c in m_l.iteritems():
            for r in c[1].iloc[:end_row_l+1].iteritems():
                for pow_1 in range(w_l):
                    if r[0] == c[0]:
                        offset = pow_1
                    else:
                        offset = 0
                    for pow_2 in range(offset, w_l):
                        coeff = pow(2, pow_1)*pow(2, pow_2)
                        # differentiating between variance and covariance
                        if (r[0]==c[0]) and (pow_1==pow_2):
                            l_quad[(r[0] + "_" +str(pow_1), c[0] + "_" + str(pow_2))] = np.sign(m_l.loc[r[0], c[0]])*c2*coeff*pow(abs(m_l.loc[r[0], c[0]]), e2)
                        else:
                            l_quad[(r[0] + "_" +str(pow_1), c[0] + "_" + str(pow_2))] = np.sign(m_l.loc[r[0], c[0]])*c3*coeff*pow(abs(m_l.loc[r[0], c[0]]), e3)
            end_row_l+=1
            
    # this constructs the quadratic term for short positions if short positions are to be included in the portfolio
    if shorts:
        for c in m_s.iteritems():
            for r in c[1].iloc[:end_row_s+1].iteritems():
                for pow_1 in range(w_s):
                    if r[0] == c[0]:
                        offset = pow_1
                    else:
                        offset = 0
                    for pow_2 in range(offset, w_s):
                        coeff = pow(2, pow_1)*pow(2, pow_2)
                        if (r[0]==c[0]) and (pow_1==pow_2):
                            # if short positions shouldn't be weighted, only letting positions with weight 1 to be taken
                            if w_shorts or (pow_1==0 and pow_2==0):
                                s_quad[(r[0] + "_-" +str(pow_1), c[0] + "_-" + str(pow_2))] = np.sign(m_s.loc[r[0], c[0]])*cs2*coeff*pow(abs(m_s.loc[r[0], c[0]]), es2)
                        # differentiating between variance and covariance
                        else:
                            if w_shorts or (pow_1==0 and pow_2==0):
                                s_quad[(r[0] + "_-" +str(pow_1), c[0] + "_-" + str(pow_2))] = np.sign(m_s.loc[r[0], c[0]])*cs3*coeff*pow(abs(m_s.loc[r[0], c[0]]), es3)
            end_row_s+=1

    quad["l"]=l_quad
    quad["s"]=s_quad
        
    del m_l, m_s
            
    return quad

def construct_linear_term_bin(return_vals_l, return_vals_s, cs1, c1, es1, e1, w_l, w_s, ratio, longs, signal_weights_l, signal_weights_s, w_shorts):
    """
    construct_linear_term_bin constructs the linear term for the binary model of our backtest
    
    :param return_vals_l: return_vals_l is a dictionary containing stock names as keys, and the return generated by those stocks
    
    :param w: specifies the number of bits to be used in the expansion of all terms
    
    :param ratio: the factor by which the returns for short positions will be scaled. If this value is equal to zero, it
    will be assumed that the portfolio only contains long positions
    
    :param signal_weights: a dictionary containing asset names as keys and the weights which will be added to their respective 
    linear terms as values. Weights will be computed based on the strength of their relationship with the given indicator
    variables, and recent changes in the indicator variables themselves. For example, this means that assets that have a strong
    relationship with indicator variables will be favored when positive returns are observed in those indicator variables
    
    :return: returns the linear term to be used in the binary backtest model, which is returned in the form of a dictionary
    """
    linear_terms = dict({})
    
    l_linear = dict()
    s_linear = dict()
    
    if longs:
        for k in return_vals_l.keys():
            for p in range(w_l):
                l_linear[k + "_" + str(p)] = -1*math.pow(2, p)*np.sign(return_vals_l[k])*pow(abs(return_vals_l[k]), e1)*c1
                if len(signal_weights_l):
                    l_linear[k + "_" + str(p)] -= math.pow(2, p)*signal_weights_l[k]

    if ratio:
        for k in return_vals_s.keys():
            for p in range(w_s):
                # if shorts shouldn't be weighted, only allow for short positions with weight 1 to be taken, which is when p==0
                if w_shorts or (p==0):
                    s_linear[k + "_-" + str(p)] = ratio*math.pow(2, p)*np.sign(return_vals_s[k])*pow(abs(return_vals_s[k]), es1)*cs1
                    
                    if len(signal_weights_s):
                        s_linear[k + "_-" + str(p)] += math.pow(2,p)*signal_weights_s[k]

    linear_terms["l"]=l_linear
    linear_terms["s"]=s_linear
    
    return linear_terms

# todo mert, partitioning, spend all?, long short ratio
def invest_binary(names, budget, net):
    """
    invest_binary calculates the money to be spent on each stock.

    :param names: an array containing names of stocks which the model has decided to invest in

    :param budget: the available capital to invest in the given stocks

    :return:  a dictionary containing unique stock names as keys (no AAPL_0 or AAPL_1 just AAPL), and the amount of money to 
    be spent on that stock as values.
    """
    unique_stocks = dict({})
    longed = set({})
    shorted = set({})
    # initialize portfolio size
    if len(names) != 0:
        p_size = 0
        total = 0

        for s in names:

            # getting the stock name, and the power to exponentiate 2 by, example: get 'AAPL' and '3' from 'AAPL_3',
            # which translates to buy math.pow(2,3) = 8 stocks of Apple
            stock_name, s_pow = s.split('_')
            if s_pow[0] != "-":
                num_stocks = math.pow(2, int(s_pow))
                longed.add(stock_name)
                stock_name = "long " + stock_name
            else:
                num_stocks = math.pow(2, -1*int(s_pow))
                shorted.add(stock_name)
                stock_name = "short " + stock_name
            
            total += num_stocks
            # collect actual unique stock names for records, so that 'AAPL_1' and 'AAPL_2' aren't both kept

            if stock_name in unique_stocks:
                unique_stocks[stock_name] += num_stocks
            else:
                unique_stocks[stock_name] = num_stocks
        if net:
            for stock_name in shorted:
                if stock_name in longed:
                    if unique_stocks["long " + stock_name] == unique_stocks["short " + stock_name]:
                        total -= (unique_stocks["long " + stock_name] + unique_stocks["short " + stock_name])
                        del unique_stocks["long " + stock_name]
                        del unique_stocks["short " + stock_name]
                    else:
                        if unique_stocks["long " + stock_name] > unique_stocks["short " + stock_name]:
                            total -= unique_stocks["short " + stock_name]
                            unique_stocks["long " + stock_name] -= unique_stocks["short " + stock_name]
                            del unique_stocks["short " + stock_name]
                        else:
                            if unique_stocks["long " + stock_name] < unique_stocks["short " + stock_name]:
                                total -= unique_stocks["long " + stock_name]
                                unique_stocks["short " + stock_name] -= unique_stocks["long " + stock_name]
                                del unique_stocks["long " + stock_name]
        for u in unique_stocks.keys():
            unique_stocks[u] = (unique_stocks[u]/total)*budget
    
    return unique_stocks

def make_disc(port, cap, prices, c_date, prior_port):
    ideal_invested_l, ideal_invested_s = 0, 0
    invested_amount_l, invested_amount_s = 0, 0
    closed_pos = []
    for pos in port:
        pos_type, stock_name = pos.split()
        if pos in prior_port:
            investable_amount = prior_port[pos] + (((port[pos]-prior_port[pos])//prices.loc[c_date, stock_name])*prices.loc[c_date, stock_name])
        else:
            investable_amount = (port[pos]//prices.loc[c_date, stock_name])*prices.loc[c_date, stock_name]
        if pos_type=="long":
            invested_amount_l+=investable_amount
            ideal_invested_l+=port[pos]
        if pos_type=="short":
            invested_amount_s+=investable_amount
            ideal_invested_s+=port[pos]
        #invested_amount+=investable_amount
        if investable_amount:
            port[pos]=investable_amount
        else:
            closed_pos.append(pos)
    for cp in closed_pos:
        del port[cp]
    port["cash_l"] = ideal_invested_l - invested_amount_l
    port["cash_s"] = ideal_invested_s - invested_amount_s
    
    port["cash_rem"]= cap-(invested_amount_l+invested_amount_s)-(port["cash_l"]+port["cash_s"])
    return port
    

def get_params(models, model_type, window_pct, cur_market):
    """
    get_params provides the parameters as estimated by the machine learning models provided
    
    :params models: a dictionary containing all the necessary machine learning models
    
    :params model_type: a list containing strings which represent the combination of machine learning models to be used
    
    :params window_pct: a dataframe representing daily percentage changes of assets included in the portfolio to be provided as an observation
    to the machine learning models
    
    :return: parameters as estimated by machine learning models
    """
    
    t_obs = []
    vols = []
    ups = 0
    for s in window_pct:
        p_val = np.mean(window_pct[s])
        vol_val = np.var(window_pct[s])
        t_obs.append(p_val)
        vols.append(vol_val)
        if p_val > 0:
            ups+=1
    ups /= len(t_obs) 
    obs = [np.mean(t_obs), np.std(t_obs), ups/len(t_obs), np.mean(vols), np.std(vols)]
    
    d_obs = []
    dv_obs = []
    for s2 in range(1,len(window_pct)):
        vol_day = np.var(window_pct.iloc[s2].values)
        avg_day = np.mean(window_pct.iloc[s2].values)
        d_obs.append(avg_day)
        dv_obs.append(vol_day)

    #print(obs)
    
    
    
    if "no_parent_longs"==model_type:
        action_index = models["no_parent_longs"][0].compute_action(obs)
        actual_action = models["no_parent_longs"][1][action_index]
        return actual_action
    
    if "no_parent_shorts"==model_type:
        action_index = models["no_parent_shorts"][0].compute_action(obs)
        actual_action = models["no_parent_shorts"][1][action_index]
        return actual_action

    if "short_condition"==model_type:
        mc=mccllelan_indicator(cur_market, 29, 39, 0.1, 0.05)
        mc_cond = 0
        mc_val = 0
        if mc is not None:
            mc_val=mc[-1]

        c_obs = [np.mean(t_obs), ups/len(t_obs), mc_val]

        c_action = models["short_condition"].compute_action(c_obs)
            
        return c_action
    
    if "long_condition"==model_type:
        mc=mccllelan_indicator(cur_market, 29, 39, 0.1, 0.05)
        mc_cond = 0
        if mc is not None:
            if mc[-1] < 0.0001:
                mc_cond = 1

        c_obs = [int(np.mean(t_obs)>0), int(ups/len(t_obs)>0.5), mc_cond]

        c_action = models["long_condition"].compute_action(c_obs)
            
        return c_action
    
    if "ratio"==model_type:
        r_action = models["ratio"].compute_action([np.mean(t_obs), np.std(t_obs), np.mean(vols), np.std(vols), np.mean(d_obs), np.std(d_obs), np.mean(dv_obs), np.std(dv_obs)])
        actual_action = ((2*r_action))/10
        return actual_action

    if "single_parent"==model_type:
        model_taken = models["s_params"][0].compute_action(obs)+1
        action_index = models["s_params"][model_taken].compute_action(obs)
        cur_params = models["s_params"][-1][model_taken-1][action_index]
        
        print("single")
        print(cur_params)
        
        return cur_params

    if "multiple_parents"==model_type:
        w=models["m_params"][0].compute_action(obs)+1
        ret_params[3] = w
        print(w)
        model_taken = models["m_params"][w][0].compute_action(obs)+1
        print(model_taken)
        action_index = models["m_params"][w][model_taken].compute_action(obs)
        print(action_index)
        cur_params = models["m_params"][-1][model_taken-1][action_index]
        #print(w, model_taken, action_taken, cur_params)
            
        print("multiple")
        print(cur_params)
        
        return cur_params

def get_signal_weights(signals_window_returns_l, signals_window_pct_l, signals_window_returns_s, signals_window_pct_s, window_pct_l, window_pct_s):
    # scaling returns generated by indicators by the same factor used on stock returns
    for s in signals_window_returns_l:
        signals_window_returns_l[s] = np.sign(signals_window_returns_l[s])*pow(abs(signals_window_returns_l[s]), e1)*c1

    for s in signals_window_returns_s:
        signals_window_returns_s[s] = np.sign(signals_window_returns_s[s])*pow(abs(signals_window_returns_s[s]), es1)*cs1

    # computing signal weights which will be added to the linear term variables
    for c in window_pct_l.columns:
        for s in signals_window_pct_l.columns:
            if c in signal_weights_l:
                signal_weights_l[c] += (signals_window_pct_l[s].cov(window_pct_l[c])*signals_window_returns_l[s])/signals_window_pct_l[s].var()
            else:
                signal_weights_l[c] = (signals_window_pct_l[s].cov(window_pct_l[c])*signals_window_returns_l[s])/signals_window_pct_l[s].var()

    for c in window_pct_s.columns:
        for s in signals_window_pct_s.columns:
            if c in signal_weights_s:
                signal_weights_s[c] += (signals_window_pct_s[s].cov(window_pct_s[c])*signals_window_returns_s[s])/signals_window_pct_s[s].var()
            else:
                signal_weights_s[c] = (signals_window_pct_s[s].cov(window_pct_s[c])*signals_window_returns_s[s])/signals_window_pct_s[s].var()
                
    return signal_weights_l, signal_weights_s

def anneal_port(linear, quad, sweeps, reads):
    # forming a binary quadratic model for the long portion of the portfolio
    bqm_upper = dimod.BinaryQuadraticModel(linear, quad, 0.0, dimod.BINARY)

    # sampler for upper triangular with coefficient 1
    sampler = neal.SimulatedAnnealingSampler()

    # running the sampler
    answer = sampler.sample(bqm_upper, num_sweeps=sweeps, num_reads=reads)

    # find the stock strategy with the lowest energy
    lowest_energy = np.inf
    best_stock_strat = []
    for r in range(len(answer.record)):
        if answer.record[r][1] < lowest_energy:
            lowest_energy = answer.record[r][1]
            best_stock_strat = answer.record[r][0]
    
    # finding the names of the stocks included in the best strategy
    best_stock_names = []
    for i in range(len(best_stock_strat)):
        cur_asset = answer.variables[i]
        if best_stock_strat[i] == 1: 
            best_stock_names.append(cur_asset)
            
    return answer, best_stock_strat, best_stock_names, lowest_energy

def filter_shorts(sep_net, short_market, best_stock_names_l, s_linear, s_quad):
    stock_lookup = set()
    # getting the names of stocks that have long positions
    if sep_net:
        stock_lookup = stock_lookup.union(set({s.split('_')[0] for s in best_stock_names_l}))

    mod_s_linear = dict()
    mod_s_quad = dict()

    # rebuilding linear and quadratic terms according to the list of stocks that may have a short position be taken on them
    for sl in s_linear:
        if(not (sl.split('_')[0] in stock_lookup)) and ((sl.split('_')[0] in short_market) or not len(short_market)):
            mod_s_linear[sl] = s_linear[sl]

    for sq in s_quad:
        if (not (sq[0].split('_')[0] in stock_lookup)) and (not (sq[1].split('_')[0] in stock_lookup)) and (((sq[0].split('_')[0] in short_market)) and ((sq[1].split('_')[0] in short_market)) or not len(short_market)):
            mod_s_quad[sq] = s_quad[sq]
        
    return mod_s_linear, mod_s_quad

def filter_longs(long_market, l_linear, l_quad):
    stock_lookup = set()
    # getting the names of stocks that have long positions

    mod_l_linear = dict()
    mod_l_quad = dict()

    # rebuilding linear and quadratic terms according to the list of stocks that may have a short position be taken on them
    for ll in l_linear:
        if(not (ll.split('_')[0] in stock_lookup)) and ((ll.split('_')[0] in long_market) or not len(long_market)):
            mod_l_linear[ll] = l_linear[ll]

    for lq in l_quad:
        if (not (lq[0].split('_')[0] in stock_lookup)) and (not (lq[1].split('_')[0] in stock_lookup)) and (((lq[0].split('_')[0] in long_market)) and ((lq[1].split('_')[0] in long_market)) or not len(long_market)):
            mod_l_quad[lq] = l_quad[lq]
        
    return mod_l_linear, mod_l_quad


def impossible_pos_gen(a_check, prev_port, cur_port, first_day, d_checklist, port_gains):
    # impossible_pos represents the positions which will be dropped from  the anneal,
    # un_ex_sell represents the positions that couldn't be closed despite not being included in the current portfolio
    # forced_hold represents the positions that had to be held due to unexecutable  trades
    # imp_cap is the capital asssociated with the positions which had to be dropped from the anneal, sell_cap is the capital
    # associated with the positions which couldn't be sold
    # d_checklist is used to check to ensure that capital associated with a stock isn't removed multiple times
    impossible_pos, un_ex_sell, forced_hold = dict(), dict(), dict()
    imp_cap, sell_cap = 0, 0
    
    if len(prev_port):
        for a_s in a_check:
            long_key = "long "+a_s
            short_key = "short "+a_s
            
            # checking to see if any new order on long positions triggered unexecutable trades
            if (long_key in prev_port) and (long_key in cur_port) and (not (a_s in d_checklist)):
                # adjusting the value of the previous day's position to account for the daily changes in the market
                adj_stock_val = prev_port[long_key] + port_gains[long_key]
                
                # increasing long position size when a ceiling was hit
                ceiling_cond_l = (cur_port[long_key] > adj_stock_val and a_check[a_s]=="ceiling")
                
                # decreasing long position size when a floor was hit
                floor_cond_l = (cur_port[long_key] < adj_stock_val and a_check[a_s]=="floor")
                
                if ceiling_cond_l or floor_cond_l:
                    print(a_s + " cannot be " + ("bought" if ceiling_cond_l else "sold") +", go back and re-anneal")
                    impossible_pos[a_s] = a_check[a_s]
                    forced_hold[long_key] = adj_stock_val
                    imp_cap += adj_stock_val
                    d_checklist.add(a_s)
                    
            if (not (long_key in prev_port)) and (long_key in cur_port) and (not (a_s in d_checklist)) and (a_check[a_s]=="ceiling"):
                print(a_s + " cannot be bought, go back and re-anneal")
                impossible_pos[a_s] = a_check[a_s]
                d_checklist.add(a_s)
                
            # checking to see if any new order on short positions triggered unexecutable trades
            if (short_key in prev_port) and (short_key in cur_port) and (not (a_s in d_checklist)):
                # adjusting the value of the previous day's position to account for the daily changes in the market
                adj_stock_val = prev_port[short_key] + port_gains[short_key]
                
                # decreasing short position when a ceiling was hit
                ceiling_cond_s = (adj_stock_val > cur_port[short_key] and a_check[a_s]=="ceiling")
                
                # increasing short position when a ceiling was hit
                floor_cond_s = (adj_stock_val < cur_port[short_key] and a_check[a_s]=="floor")
                
                if ceiling_cond_s or floor_cond_s:
                    print(a_s + " cannot be "+ ("bought" if ceiling_cond_s else "sold") +", go back and re-anneal")
                    impossible_pos[a_s] = a_check[a_s]
                    forced_hold[short_key] = adj_stock_val
                    imp_cap += adj_stock_val
                    d_checklist.add(a_s)
            
            if (not (short_key in prev_port)) and (short_key in cur_port) and (not (a_s in d_checklist)) and (a_check[a_s]=="floor"):
                print(a_s + " cannot be sold, go back and re-anneal")
                impossible_pos[a_s] = a_check[a_s]
                d_checklist.add(a_s)
                
        # if no impossible orders were placed, checking to see if previously existing positions can be closed, and holding onto them otherwise
        if not len(impossible_pos):
            for a_s in a_check:
                long_key = "long "+a_s
                short_key = "short "+a_s
                
                # checking to see if we attempted to close a long position even though that stock hit a floor
                if long_key in prev_port and (not long_key in cur_port) and a_check[a_s]=="floor" and (not (a_s in d_checklist)):
                    print(a_s + " cannot be sold, go back and redistribute")
                    adj_stock_val = prev_port[long_key] + port_gains[long_key]
                    un_ex_sell[a_s] = a_check[a_s]
                    sell_cap += adj_stock_val
                    forced_hold[long_key] = adj_stock_val
                    d_checklist.add(a_s)
                    
                # checking to see if we attempted close a short position even thought that stock hit a ceiling
                if short_key in prev_port and (not short_key in cur_port) and a_check[a_s]=="ceiling" and (not (a_s in d_checklist)):
                    adj_stock_val = prev_port[short_key] + port_gains[short_key]
                    print(a_s + " cannot be bought, go back and redistribute")
                    un_ex_sell[a_s] = a_check[a_s]
                    sell_cap += adj_stock_val
                    forced_hold[short_key] = adj_stock_val
                    d_checklist.add(a_s)
    else:
        # this is for the case of the first day of trading and the previous day having an empty portfolio, during which there are no previously 
        # held positions
        for a_s in a_check:
            long_key = "long "+a_s
            short_key = "short "+a_s
            if long_key in cur_port and a_check[a_s]=="ceiling":
                impossible_pos[a_s] = "ceiling"
            if short_key in cur_port and a_check[a_s]=="floor":
                impossible_pos[a_s] = "floor"
            
    return impossible_pos, un_ex_sell, imp_cap, sell_cap, forced_hold, d_checklist

# quick note: ema_cond, s_l_ratio, and apriori_ratio are all various ways of penalizing/allowing the taking of short positions which all exist due to the
# many iterations our model went through and will eventually be combined in a way that makes sense
def back_test_binary_ml(historical_data, cs1=1, cs2=1, cs3=1, c1=1, c2=1, c3=1, e1=1, e2=1, e3=1, es1=1, es2=1, es3=1, window_l=10, window_s=10, sweeps=100, reads=1000, capital=10000, w_l=1, w_s=1, s_l_ratio=1, ema_smooth_l=2, ema_smooth_s=2, longs=True, signals=pd.DataFrame(), net=False, w_shorts=True, progress_file="", output_file="", ret_vals=dict(), models=False, model_type=["short_condition"], sep_net=True, train=False, index_df=pd.DataFrame(), start_date="", apriori_ratio=0.2, long_hist=pd.DataFrame(), short_market=set(), graph=True, rf_tol=0.096, last_port_rf={}, vol_norm=False, diff_prev_ar="no", hist_t_prices=pd.DataFrame(), long_market=set(), disc=False):
    """
    back_test_binary conducts a backtest using the binary weighted scheme
    
    :param historical data: the historical data on which trades are conducted, provided in the form of a dataframe
    
    :param c1: the factor by which returns are multiplied, set to 1 by default
    
    :param c2: the factor by which variances are multiplied, set to 1 by default
    
    :param c3: the factor by which covariances are multiplied, set to 1 by default
    
    :param e1: the factor by which returns are exponentiated (sign preserving), set to 1 by default
    
    :param e2: the factor by which variances are exponentiated (sign preserving), set to 1 by default
    
    :param e3: the factor by which covariances are exponentiated (sign preserving), set to 1 by default
    
    :param window: the number of days used in training our model which is set to 10 by default
    
    :param sweeps: number of sweeps used by the annealing model
    
    :param reads: number of samples taken by the annealing model
    
    :param capital: the amount of capital available to be invested during the backtest
    
    :param w: the number of bits to be used in the bit expansion of the budget term
    
    :param s_l_ratio: the factor by which the returns for short positions will be scaled. If this value is equal to zero, it
    will be assumed that the portfolio only contains long positions
    
    :param longs: a boolean variable which allows long positions to be taken if set to True, and prevents long positions from
    being taken if set to False
    
    :param signals: a pandas dataframe containing historical price information for indicator variables. The assets contained
    within this dataframe will not be included within the portfolio, but instead act as indicators to signal when it would be
    the best time to enter long or short positions in particular assets
    
    :param new_lin: a boolean variable which uses a method involving adding an exponentially scaling constant to short position
    components to adjust scaling between short and long positions if set to True (was experimental, and is no longer being used)
    
    :param w_shorts: a boolean variable which allows the short positions in the portfolio to be weighted. If set to False, every
    short position can only receive a weight of 1.

    :param progress_file: a stirng variable representing the name of the csv file which will be modified to record progress
    
    :param output_file: a string variable representing the name of the csv file which will be generated at the end of the 
    backtest
    
    :param ret_vals: a dictionary which would be used to return backtest results if multiprocessing is being used. If
    multiprocessing is not being used, do not pass a value to this variable
    
    :param models: If the user passes an array containing Machine Learning Models, the hyperparameters will be generated each
    day using the given models. Otherwise, the hyperparameters used will be constant and based on the values passed to the
    parameters when this function was called
    
    :param separate: a boolean varaible which determines whether separate annealers should be used when optimizing long and
    short positions
    
    :param sep_net: a boolean variable which determines whether stocks which were picked when long positions were being
    optimized should be considered when short positions are being optimized. This parameter is only of consequence if separate
    is set to True, otherwise the value of this parameter willl be of no impact
    
    :param train: a boolean variable which determines whether the provided machine learning models should continue to be trained
    throughout the duration of the backtest. This means that after each day of backtesting, the model will be trained with an
    additional sample. If machine learning models are not provided, the value of this parameter will be of no effect.
    
    :return: an array of tuples where the first element of the tuple is a date, and the second element of the tuple is a list
    of the tickers included in the portfolio that day
    """
    
    #
    daily_portfolio = []
    if len(last_port_rf):
        if isinstance(diff_prev_ar, str):
            prev_ar = apriori_ratio
            print("static ar: ", prev_ar)
        else:
            prev_ar = diff_prev_ar
            print("dynamic ar: ", prev_ar)
        prior_cap = sum(last_port_rf["portfolio"].values())
        if not prior_cap:
            prior_cap = capital
        daily_portfolio.append(["TEMP", last_port_rf["portfolio"], len(last_port_rf["portfolio"]), np.nan, np.nan, capital, np.nan, np.nan, np.nan, last_port_rf["long_cum"], np.nan, last_port_rf["short_cum"]])
        empty = False
        if (not len(last_port_rf["portfolio"])) or (len(last_port_rf["portfolio"])==3 and ("cash_l" in last_port_rf["portfolio"]) and ("cash_s" in last_port_rf["portfolio"]) and ("cash_rem" in last_port_rf["portfolio"])):
            empty = True
    if len(historical_data) < min(window_l, window_s):
        print("The amount of data provided does not contain sufficient data for the window provided")
    else:
        data = historical_data.copy(deep=True)
        data = data.replace(0, np.nan)
        
        # keeping a table without any dropped stocks to ensure stocks which were included in the previous day's portfolio which were now dropped
        # don't lead to any issues
        historical_gain_table = data.copy(deep=True)
        historical_gain_table = rep_nan(historical_gain_table)
        historical_gain_pct = historical_gain_table.pct_change()
        # data cleaning
        #data = drop_vol(data)
        
        # replace missing t_prices with adj closes, and match t_prices with close data
        if len(hist_t_prices):
            t_prices = pd.DataFrame(columns=data.columns)
            for hist_ind in data.index:
                for hist_c in data.columns:
                    if hist_ind in hist_t_prices.index and hist_c in hist_t_prices.columns:
                        price_from_t = hist_t_prices.loc[hist_ind, hist_c]
                        if price_from_t and not np.isnan(price_from_t):
                            t_prices.loc[hist_ind, hist_c] = price_from_t
                        else:
                            #print(hist_ind, hist_c)
                            t_prices.loc[hist_ind, hist_c] = data.loc[hist_ind, hist_c]
                    else:
                        t_prices.loc[hist_ind, hist_c] = data.loc[hist_ind, hist_c]
                        
            historical_gain_table_t = t_prices.copy(deep=True)
            #historical_gain_table_t = historical_gain_table_t.replace(0, np.nan)
            historical_gain_table_t = rep_nan(historical_gain_table_t)
            historical_gain_pct_t = historical_gain_table_t.pct_change()
            
            t_prices = remove_assets(t_prices)
            t_prices = rep_nan(t_prices)
                        
        data = remove_assets(data)
        data = rep_nan(data)
        #print("SUCCESS")
        #return data, t_prices
        # if previously generated data for long positions is provided, it will not be necessary to anneal long positions again
        if len(long_hist):
            longs=False
        
        if len(signals):
            
            # making sure that the indexes of the stock price dataframe and indicator dataframe match, and removing any non
            # matching indexes from the indicator dataframe
            new_indic = pd.DataFrame(index=data.index, columns=signals.columns)
            for c in signals.columns:
                for i in data.index:
                    # currently in here because some of our data sources store dates as datetime objects which include 
                    # hours/minutes/seconds, this code ensures that only month/day/year information is included
                    date_str = str(i).split()[0]
                    if date_str in signals.index:
                        new_indic.at[date_str,c] = signals.loc[date_str,c]
            
            signals_df = new_indic.copy(deep=True)
            
            # data cleaning for the indicators
            signals_df = remove_assets(signals_df)
            signals_df = rep_nan(signals_df)
            
            # currently commented out for testing purposes
            #signals_df = drop_vol(signals_df)
            
            # computing percentage changes for the indicator variables
            signals_pct = signals_df.pct_change()
            
            print(signals_df)
        
        # compute the percentage changes for the entire dataset
        pct_changes = data.pct_change()
        if len(hist_t_prices):
            pct_changes_t_prices = t_prices.pct_change()
        #historical_gain_pct_t = historical_gain_pct_t.drop(["KCAER.IS", "KLRHO.IS"], axis=1)
        #print("t equals ", historical_gain_pct_t.equals(pct_changes_t_prices))
        #historical_gain_pct = historical_gain_pct.drop(["KCAER.IS", "KLRHO.IS"], axis=1)
        #print("gen equals", pct_changes.equals(historical_gain_pct))
        #print(1/0)

        
        # initializing the start and end points of the window
        counter = 0
        if len(last_port_rf):
            counter = 1
        ema_vals_l, ema_vals_s = dict(), dict()

        end_point = max(window_l-1, window_s-1)
        start_point_l, start_point_s = end_point-(window_l-1), end_point-(window_s-1)
        
        if len(start_date):
            # getting the index of the first day that will be considered in the anneal process, i.e the latest day to be included in linear and quadratic terms 
            # for the first window
            start_index = data.index.get_loc(start_date)
            
            end_point = start_index
            
            # getting individual start sizes for long and short terms to accomodate for different window sizes
            start_point_l=start_index-(window_l-1)
            start_point_s=start_index-(window_s-1)
            
            init_start_s, init_end_s = 0, window_s-1
            init_start_l, init_end_l = 0, window_l-1
            
            # the place_holder term below just acts as a placeholder for a term that would otherwise be used in th construction of the linear term, which is not 
            # necessary when initializing the ema values
            
            # ema values are initialized separately to accomodate for different window sizes
            
            # initializing ema values for long positions
            for i_day_l in range(init_end_l, end_point):
                init_cur_window_l = data.iloc[init_start_l:i_day_l+1]
                ema_vals_l, place_holder = ema(init_cur_window_l, ema_vals_l, window_l, ema_smooth_l)
                init_start_l+=1
            
            # initializing ema_values for short positions
            for i_day_s in range(init_end_s, end_point):
                init_cur_window_s = data.iloc[init_start_s:i_day_s+1]
                ema_vals_s, place_holder = ema(init_cur_window_s, ema_vals_s, window_s, ema_smooth_s)
                init_start_s+=1
        empty_cap, empty_long_size, empty_short_size = -1, -1, -1
        # day represents the latest day which will be included as part of the linear and quadratic terms
        for day in tqdm(range(end_point, len(data))):
            start_time = time.time()
            counter += 1
            
            # index data which includes all available data up until the latest day. This is primarily used as part of indicator
            cur_market = index_df.loc[:str(data.index[day]).split()[0]]
            
            # get data for the current windows
            current_window_l, current_window_s = data.iloc[start_point_l:day+1], data.iloc[start_point_s:day+1]
            
            # if we're doing a grid search with t prices, replace last day with t prices for anneal
            t_date_used = current_window_l.index[-1]
            if len(hist_t_prices):
                for c_l in current_window_l.columns:
                    current_window_l.loc[t_date_used, c_l] = t_prices.loc[t_date_used, c_l]
                for c_s in current_window_s.columns:
                    current_window_s.loc[t_date_used, c_s] = t_prices.loc[t_date_used, c_s]

            # getting signal data from window
            if len(signals):
                signals_window_l, signals_window_s = signals_df.iloc[start_point_l:day+1], signals_df.iloc[start_point_s:day+1]

            # get percentage changes for the current windows
            window_pct_l, window_pct_s = pct_changes.iloc[start_point_l:day+1], pct_changes.iloc[start_point_s:day+1]
            
            # replacing pct change value in last day with pct change between yesterday's close and today's t price
            if len(hist_t_prices):
                t_l_pct = current_window_l.pct_change()
                t_s_pct = current_window_s.pct_change()
                for l_pct in t_l_pct.columns:
                    window_pct_l.loc[t_date_used, l_pct] = t_l_pct.loc[t_date_used, l_pct]
                for s_pct in t_s_pct.columns:
                    window_pct_s.loc[t_date_used, s_pct] = t_s_pct.loc[t_date_used, s_pct]
                
            
            # NEW: checking if stocks have hit either a floor or a ceiling in which case they would be unexecutable
            anneal_check = dict()
            for r_stock in window_pct_l.columns:
                if window_pct_l.iloc[-1][r_stock] > rf_tol:
                    anneal_check[r_stock] = "ceiling"
                if window_pct_l.iloc[-1][r_stock] < -1*rf_tol:
                    anneal_check[r_stock] = "floor"
            
            # getting percentage changes for the current signal windows
            if len(signals):
                signals_window_pct_l, signals_window_pct_s = signals_pct.iloc[start_point_l:day+1], signals_pct.iloc[start_point_s:day+1]
            
            # compute daily parameters using ml model - work in progress
            """
            if models:
                if "short_condition" in model_type:
                    condition_action = get_params(models, "short_condition", window_pct_s, cur_market)
                    s_l_ratio = s_l_ratio_copy*condition_action
                    #apriori_ratio = apriori_copy*condition_action
                    ema_cond = condition_action
                    
                if "long_condition" in model_type:
                    condition_action = get_params(models, "long_condition", window_pct_l, cur_market)
                    longs = condition_action
                    
                if "ratio" in model_type:
                    apriori_ratio = get_params(models, "ratio", window_pct_s, cur_market)
                
                if "no_parent_longs" in model_type:
                    c1 = get_params(models, "no_parent_longs", window_pct_l, cur_market)
                
                if "no_parent_shorts" in model_type:
                    cs1 = get_params(models, "no_parent_shorts", window_pct_s, cur_market)
            """
            
            # this ensures that short positions are not taken if the apriori ratio is set to zero, this check is performed with each iteration
            # as we plan on dynamically setting the apriori ratio with each iteration in the future, even though it is currently static
            s_l_ratio = 1
            if not apriori_ratio:
                s_l_ratio = 0
            
            # compute returns and ema values for the current windows
            ema_vals_l, window_returns_l = ema(current_window_l, ema_vals_l, window_l, ema_smooth_l)
            ema_vals_s, window_returns_s = ema(current_window_s, ema_vals_s, window_s, ema_smooth_s)
            
            # calculate returns for signal windows
            if len(signals):
                signals_window_returns_l, signals_window_returns_s  = calculate_returns(signals_window_l), calculate_returns(signals_window_s)
            
            # create a covariance matrix for the current windows using percentage change data
            window_cov_l, window_cov_s = window_pct_l.cov(), window_pct_s.cov()
            
            
            # turn the covariance matrix into an upper triangular matrix
            upper_cov_l = pd.DataFrame(np.triu(window_cov_l), index=window_cov_l.index, columns=window_cov_l.columns)
            upper_cov_s = pd.DataFrame(np.triu(window_cov_s), index=window_cov_s.index, columns=window_cov_s.columns)
            if vol_norm:
                for swl in window_returns_l:
                    swl_var = upper_cov_l.loc[swl, swl]
                    if swl_var==0:
                        swl_var=1
                    window_returns_l[swl] = window_returns_l[swl]/swl_var
                for sws in window_returns_s:
                    sws_var = upper_cov_s.loc[sws, sws]
                    if sws_var==0:
                        sws_var=1
                    window_returns_s[sws] = window_returns_s[sws]/sws_var
                    

            # reformulate the upper triangular covariance matrix into the quadratic term for the binary model
            # this returns a dictionary that contains the quadratic term for both long and short positions
            quadratic_term = construct_quadratic_term_bin(upper_cov_mat_l=upper_cov_l, upper_cov_mat_s=upper_cov_s, w_l=w_l, w_s=w_s, shorts=s_l_ratio, longs=longs, cs2=cs2, cs3=cs3, c2=c2, c3=c3, es2=es2, es3=es3, e2=e2, e3=e3, w_shorts=w_shorts)
            
            # getting signal weights if a signal time series was provided
            signal_weights_l, signal_weights_s = dict({}), dict({})
            if len(signals):
                signal_weights_l, signal_weights_s = get_signal_weights(signals_window_returns_l=signals_window_returns_l, 
                                                                        signals_window_pct_l=signals_window_pct_l, 
                                                                        signals_window_returns_s=signals_window_returns_s, 
                                                                        signals_window_pct_s=signals_window_pct_s, 
                                                                        window_pct_l=window_pct_l, 
                                                                        window_pct_s=window_pct_s
                                                                       )
                    
            # constructing linear term for the binary model
            linear_term = construct_linear_term_bin(return_vals_l=window_returns_l, 
                                                    return_vals_s=window_returns_s, 
                                                    cs1=cs1, 
                                                    c1=c1, 
                                                    es1=es1, 
                                                    e1=e1, 
                                                    w_l=w_l,
                                                    w_s=w_s,
                                                    ratio=s_l_ratio, 
                                                    longs=longs, 
                                                    signal_weights_l=signal_weights_l, 
                                                    signal_weights_s=signal_weights_s, 
                                                    w_shorts=w_shorts
                                                   )
            
            # getting separate quadratic and linear terms for the long and short portions of the portfolio
            
            l_linear, s_linear= linear_term["l"], linear_term["s"]
            l_quad, s_quad =quadratic_term["l"], quadratic_term["s"]
            #print("!!")
            #print(len(l_linear),len(l_quad))
            #print("!!")
            
            re_anneal, first_attempt = True, True
            impossible_pos, un_ex_sell, forced_hold = dict(), dict(), dict()
            d_checklist = set()
            cap_modifier = 0
            print(data.index[day])
            while re_anneal:
                re_anneal = False
                
                imp_and_unex_dict = dict()
                imp_and_unex_dict.update(impossible_pos.copy())
                imp_and_unex_dict.update(un_ex_sell.copy())
                # filtering stocks based on whether an action taken on them following an anneal is allowed
                if len(imp_and_unex_dict):
                    ll_drop, ls_drop, ql_drop, qs_drop = set(), set(), set(), set()
                    for imp in imp_and_unex_dict:
                        for l_k in l_linear:
                            if imp == l_k.split("_")[0]:
                                ll_drop.add(l_k)
                        for s_k in s_linear:
                            if imp == s_k.split("_")[0]:
                                ls_drop.add(s_k)
                        for cov_kl in l_quad:
                            lp_first, lp_second = cov_kl
                            if imp==lp_first.split("_")[0] or imp==lp_second.split("_")[0]:
                                ql_drop.add(cov_kl)
                        for cov_ks in s_quad:
                            sp_first, sp_second = cov_ks
                            if imp==sp_first.split("_")[0] or imp==sp_second.split("_")[0]:
                                qs_drop.add(cov_ks)
                    for d1 in ll_drop:
                        l_linear.pop(d1)
                    for d2 in ls_drop:
                        s_linear.pop(d2)
                    for d3 in ql_drop:
                        l_quad.pop(d3)
                    for d4 in qs_drop:
                        s_quad.pop(d4)
                        
                                
                impossible_pos, un_ex_sell, imp_and_unex_dict = dict(), dict(), dict()
                
                # if the only unexecutable orders are the closing of existing positions, no need to re-anneal, just reallocate capital

                # annealing long positions and retrieving the optimal portfolio
                #print(l_linear)
                #print("****")
                #print(("DEVA.IS_0", "OYAKC.IS_0") in l_quad)
                #print(len(l_quad), len(l_linear))
                if len(long_market):
                    l_linear, l_quad = filter_longs(long_market, l_linear, l_quad)
                #print("---")
                #print(l_linear)
                #print("***")
                #print(("DEVA.IS_0", "OYAKC.IS_0") in l_quad)
                #print(len(l_quad), len(l_linear))
                answer_l, best_stock_strat_l, best_stock_names_l, lowest_energy_l = anneal_port(l_linear, l_quad, sweeps, reads)

                # if previously generated data was provided for investment decisions, get stock names from there instead
                if len(long_hist):
                    ex_day = str(data.index[day+1]).split()[0]
                    best_stock_names_l = long_hist.loc[ex_day]["Stocks"]

                # ensuring that short positions are only taken on assets without long positions, and from a specific subset of the broader market, if the necessary
                # parameters are provided
                mod_s_linear, mod_s_quad = s_linear, s_quad
                if sep_net or len(short_market):
                    mod_s_linear, mod_s_quad = filter_shorts(sep_net, short_market, best_stock_names_l, s_linear, s_quad)

                # annealing short positions and retrieving the optimal portfolio
                answer_s, best_stock_strat_s, best_stock_names_s, lowest_energy_s = anneal_port(mod_s_linear, mod_s_quad, sweeps, reads)

                # getting the lowest energy found from either portfolio for records (record both?)
                lowest_energy = min(lowest_energy_l, lowest_energy_s)

                # getting best stock names included in either portfolio
                best_stock_names = best_stock_names_l + best_stock_names_s
                actual_best_stock_names_l = set({"long " + bn.split("_")[0] for bn in best_stock_names_l})
                actual_best_stock_names_s = set({"short " + bn.split("_")[0] for bn in best_stock_names_s})
                
                
                # recording the tickers of the selected stocks in the dictionary that will be returned
                # execution_day specifies the state of the portfolio at the end of that day.
                # For example, if the portfolio has stocks A and B on execution_day 1, and stocks B and C on execution_day 2,
                # that means that stocks A and B would have been sold on execution_day 2, and stocks B and C would have
                # been purchased on execution_day 2

                execution_day = data.index[day]
                #print("Day", counter)

                # initializing the portfolio record on the first day
                if not len(daily_portfolio):
                    temp_names = actual_best_stock_names_l.copy()
                    temp_names.update(actual_best_stock_names_s)
                    impossible_pos, un_ex_sell, imp_cap, sell_cap, forced_hold_t, d_checklist = impossible_pos_gen(anneal_check, {}, temp_names, True, d_checklist, {})
                    if len(impossible_pos):
                        re_anneal = True
                    else:
                        

                        # if apriori_ratio = 0, invest all of the capital into long positions
                        if not apriori_ratio:
                            if first_attempt:
                                ideal_investments = invest_binary(best_stock_names, capital, net)
                            investments = invest_binary(best_stock_names, capital, net)
                        else:
                            # NOTE: if an empty list of stocks is provided to invest_binary, invest_binary returns an empty dictionary

                            # if short positions are taken, only invest previously set ratio into longs, otherwise invest entire portfolio into longs
                            if len(best_stock_names_s):
                                if first_attempt:
                                    ideal_investments_l = invest_binary(best_stock_names_l, capital*(1-apriori_ratio), net)
                                investments_l = invest_binary(best_stock_names_l, capital*(1-apriori_ratio), net)
                            else:
                                if first_attempt:
                                    ideal_investments_l = invest_binary(best_stock_names_l, capital, net)
                                investments_l = invest_binary(best_stock_names_l, capital, net)

                            # if long positions are taken, only invest previously set ratio into shorts, otherwise invest entire portfolio into shorts
                            if len(best_stock_names_l):
                                if first_attempt:
                                    ideal_investments_s = invest_binary(best_stock_names_s, capital*(apriori_ratio), net)
                                investments_s = invest_binary(best_stock_names_s, capital*(apriori_ratio), net)
                            else:
                                if first_attempt:
                                    ideal_investments_s = invest_binary(best_stock_names_s, capital, net)
                                investments_s = invest_binary(best_stock_names_s, capital, net)
                            if first_attempt:
                                ideal_investments_l.update(ideal_investments_s)
                                ideal_investments = ideal_investments_l.copy()
                                
                            investments_l.update(investments_s)
                            investments = investments_l.copy()
                            
                            print("after disc ", investments)
                        if disc:
                            if len(hist_t_prices):
                                investments = make_disc(investments, capital, t_prices, execution_day, {})
                            else:
                                investments = make_disc(investments, capital, data, execution_day, {})

                        # portfolio columns in order: execution date, list of investments in dictionary format 
                        # (keys are stocks and kind of position, values are position sizes), number of stocks in portfolio, lowest energy found,
                        # current capital, daily return, cumulative return, daily long return, cumulative long return, daily short return, cumulative short return
                        invest_len = len(investments)
                        if disc:
                            invest_len-=3
                            
                        empty = False
                        if not invest_len:
                            empty = True

                        daily_data = [execution_day, investments, invest_len, lowest_energy, time.time()-start_time, capital, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                        daily_portfolio.append(tuple(daily_data))
                        # record new information in a csv file if a name is specified
                        if len(progress_file):
                            update_progress(progress_file, execution_day, daily_data)
                else:
                    port_gains = dict()
                    if not empty:
                        # sell portfolio from previous day and compute portfolio value
                        prev_port = daily_portfolio[counter-2][1]
                        long_size, short_size, p_size = 0, 0, 0
                        
                        if empty_cap != -1:
                            print("empty cap triggered on day", execution_day)
                            p_size = empty_cap
                        # these variables are used to quickly check whether long/short positions were taken in the previous portfolio
                        long_prev, short_prev = False, False
                        prev_long_alloc, prev_short_alloc = 0, 0
                        print("prev port invested size: ", sum(prev_port.values()), execution_day)
                        for s in prev_port:
                            if not (s in ["cash_l", "cash_s", "cash_rem"]):
                                action, stock = s.split()

                                sign = 1
                                if action == "short":
                                    sign = -1
                                    short_prev = True
                                    prev_short_alloc+=prev_port[s]
                                else:
                                    long_prev = True
                                    prev_long_alloc+=prev_port[s]
                                if len(hist_t_prices):
                                    gains = prev_port[s]*(1+sign*historical_gain_pct_t[stock][execution_day])
                                    port_gains[action + " " + stock] = prev_port[s]*(sign*historical_gain_pct_t[stock][execution_day])
                                else:
                                    gains = prev_port[s]*(1+sign*historical_gain_pct[stock][execution_day])
                                    port_gains[action + " " + stock] = prev_port[s]*(sign*historical_gain_pct[stock][execution_day])

                                # computing total current overall portfolio size along with long and short gains
                                p_size += gains
                                long_size += gains*int(action=="long")
                                short_size += gains*int(action=="short")
                            else:
                                p_size+=prev_port[s]

                    else:
                        # if no actions were taken the previous day, take last day's portfolio size values
                        p_size = daily_portfolio[counter-2][5]

                    # if apriori_ratio = 0, invest all of the capital into long positions
                    if not apriori_ratio:
                        if first_attempt:
                            ideal_investments = invest_binary(best_stock_names, p_size, net)
                        print("capital available: " + str(p_size), "capital invested: " + str(p_size-cap_modifier))
                        print("cap modifier: " +  str(cap_modifier))
                        investments = invest_binary(best_stock_names, p_size-cap_modifier, net)
                        print("investments before update: ", investments)
                    else:
                        if len(best_stock_names_s):
                            if first_attempt:
                                ideal_investments_l = invest_binary(best_stock_names_l, p_size*(1-apriori_ratio), net)
                            investments_l = invest_binary(best_stock_names_l, (p_size-cap_modifier)*(1-apriori_ratio), net)
                        else:
                            if first_attempt:
                                ideal_investments_l = invest_binary(best_stock_names_l, p_size, net)
                            
                            investments_l = invest_binary(best_stock_names_l, p_size-cap_modifier, net)
                        if len(best_stock_names_l):
                            if first_attempt:
                                ideal_investments_s = invest_binary(best_stock_names_s, p_size*(apriori_ratio), net)
                            investments_s = invest_binary(best_stock_names_s, (p_size-cap_modifier)*(apriori_ratio), net)
                        else:
                            if first_attempt:
                                ideal_investments_s = invest_binary(best_stock_names_s, p_size, net)
                            
                            investments_s = invest_binary(best_stock_names_s, p_size-cap_modifier, net)
                            
                        if first_attempt:
                            ideal_investments_l.update(ideal_investments_s)
                            ideal_investments = ideal_investments_l.copy()
                            
                        investments_l.update(investments_s)
                        investments = investments_l.copy()
                    if disc:
                        if len(hist_t_prices):
                            investments = make_disc(investments, p_size-cap_modifier, t_prices, execution_day, ({} if empty else prev_port))
                        else:
                            investments = make_disc(investments, p_size-cap_modifier, data, execution_day, ({} if empty else prev_port))
                        
                    impossible_pos, un_ex_sell, imp_cap, sell_cap, forced_hold_t, d_checklist = impossible_pos_gen(anneal_check, ({} if empty else prev_port), investments, False, d_checklist, port_gains)
                    forced_hold.update(forced_hold_t.copy())
                    
                    if len(impossible_pos) or len(un_ex_sell):
                        print("relooping")
                        print("impossible poses ", impossible_pos)
                        print("impossible sells ", un_ex_sell)
                        cap_modifier += (imp_cap + sell_cap)
                        re_anneal = True
                    else:
                        print("forced holds: ", forced_hold)
                        empty_cap = -1
                        
                        redo_cap = False
                        if not len(investments) and len(forced_hold):
                            print("redoing cap")
                            redo_cap = True
                        
                        investments.update(forced_hold.copy())
                        
                        print("investments after update: ", investments)
                        if not empty:
                            long_factor, long_factor_denom = (1-prev_ar), (1-prev_ar)
                            short_factor, short_factor_denom = prev_ar, prev_ar

                            if not short_prev:
                                long_factor, short_factor = 1, 0
                                short_factor_denom, long_factor_denom = 1, 1

                            if not long_prev:
                                short_factor, long_factor = 1, 0
                                long_factor_denom, short_factor_denom = 1, 1
                                
                            if empty_long_size!=-1 or empty_short_size!=-1:
                                print("--")
                                print("got some holds")
                                if empty_long_size!=-1:
                                    if empty_long_size:
                                        daily_long = (long_size-empty_long_size)/(daily_portfolio[counter-2][5])
                                    else:
                                        daily_long = 0
                                    empty_long_size = -1

                                if empty_short_size!=-1:
                                    if empty_short_size:
                                        print(short_size, empty_short_size)
                                        daily_short = (short_size-empty_short_size)/(daily_portfolio[counter-2][5])
                                    else:
                                        daily_short = 0
                                    empty_short_size = -1
                                    print(daily_short)
                                    print("--")
                                
                            else:
                                print("**")
                                if disc:
                                    daily_long = (long_size-prev_long_alloc)/(prev_long_alloc+prev_port["cash_l"]+(prev_port["cash_rem"]*long_factor))
                                    daily_short = (short_size-prev_short_alloc)/(prev_short_alloc+prev_port["cash_s"]+(prev_port["cash_rem"]*short_factor))
                                else: 
                                    daily_long = (long_size-(daily_portfolio[counter-2][5])*long_factor)/(daily_portfolio[counter-2][5]*long_factor_denom)
                                    daily_short = (short_size-(daily_portfolio[counter-2][5])*short_factor)/(daily_portfolio[counter-2][5]*short_factor_denom)
                                print("daily short: " + str(daily_short), "short size: " + str(short_size), "short factor: " + str(short_factor), "total portfolio size: " + str(daily_portfolio[counter-2][5]), "denom factor: " + str(short_factor_denom))
                                print("**")
                            prev_long_cum = daily_portfolio[counter-2][9]

                            if np.isnan(prev_long_cum):
                                cur_long_cum = daily_long
                            else:
                                cur_long_cum = ((1+prev_long_cum)*(1+daily_long)) - 1

                            prev_short_cum = daily_portfolio[counter-2][11]
                            if np.isnan(prev_short_cum):
                                cur_short_cum = daily_short
                            else:
                                cur_short_cum = ((1+prev_short_cum)*(1+daily_short)) - 1
                        else:
                            daily_long, daily_short = 0, 0
                            cur_long_cum, cur_short_cum = daily_portfolio[counter-2][9], daily_portfolio[counter-2][11]
                        if redo_cap:
                            empty_cap = p_size
                            empty_long_size = 0
                            empty_short_size = 0
                            for f_stock in forced_hold:
                                empty_cap -= forced_hold[f_stock]
                                if f_stock.split()[0]=="long":
                                    empty_long_size += forced_hold[f_stock]
                                else:
                                    empty_short_size += forced_hold[f_stock]
                        redo_cap = False
                        
                        invest_len = len(investments)
                        if disc:
                            invest_len-=3

                        daily_data = [execution_day, investments, invest_len, lowest_energy, time.time()-start_time,p_size,(p_size-daily_portfolio[counter-2][5])/daily_portfolio[counter-2][5],(p_size-daily_portfolio[0][5])/daily_portfolio[0][5], daily_long, cur_long_cum, daily_short, cur_short_cum]
                        daily_portfolio.append(tuple(daily_data))
                        if len(progress_file):
                            update_progress(progress_file, execution_day, daily_data)

                        # temporary print statement for visualization

                        empty = False
                        if not invest_len:
                            empty = True
                        
                first_attempt = False    
            prev_ar = apriori_ratio
            
            start_point_l += 1
            start_point_s += 1
            if len(daily_portfolio) > 1:
                print("current capital: ", p_size)
                print("daily change: " + str((p_size-daily_portfolio[counter-2][5])/daily_portfolio[counter-2][5]))
                #print(str(execution_day) + "," + str(investments) + "," + str(len(investments)) + "," + str(lowest_energy) + "," + str(time.time()-start_time) + "," + str(p_size) + "," + str((p_size-daily_portfolio[counter-2][5])/daily_portfolio[counter-2][5]) + "," + str((p_size-daily_portfolio[0][5])/daily_portfolio[0][5]))
            
            if graph:
                times = [x[0] for x in daily_portfolio]
                vals = [y[5] for y in daily_portfolio]
                plt.plot(times, vals)
                clear_output(wait=False)
                plt.show()
                plt.pause(0)
            print("----")
        if len(last_port_rf):
            daily_portfolio = daily_portfolio[1:]
        # output file and add final values to the dictionary for multiprocessing
        if len(output_file):
            final_port = organize_results(daily_portfolio)
            final_port.to_csv(output_file)
            ret_vals[output_file] = final_port
        
        del data
    return daily_portfolio

def ema(data, init_v, period, smoothing=2):
    """
    ema generates a pandas series representation of an exponential moving average applied on a given set of data.
    
    :param data: a pandas series representing the time series the exponential moving average function will be applied upon
    
    :param period: an integer representing the number of days to be considered when computing the exponential moving average
    
    :param smoothing: an integer representing the smoothing factor to be used by the exponential moving average and is set
    to 2 by default.
    
    :return: a pandas time series representing the exponential moving average applied on the parameter data
    """
    if smoothing < -1 and (not len(init_v)):
        ret_dict = dict()
        ret_pct = dict()
        
        ret_vals = np.mean(data.iloc[:period])
        for k in ret_vals.index.values:
            ret_dict[k] = [ret_vals[k]]
            ret_pct[k] = (ret_vals[k]-data.iloc[0][k])/(data.iloc[0][k]*period)
        return ret_dict, ret_pct
    
    if smoothing==-1:
        ret_dict = dict()
        ret_pct = dict()
        
        ret_vals = np.mean(data.iloc[:period])
        for k in ret_vals.index.values:
            ret_dict[k] = [ret_vals[k]]
            ret_pct[k] = (data.iloc[period-1][k]-data.iloc[0][k])/(data.iloc[0][k]*period)
        return ret_dict, ret_pct
    
    if (smoothing > 0 and len(init_v)==0) or not smoothing:
        ret_dict = dict()
        ret_pct = dict()
        
        ret_vals = np.mean(data.iloc[:period])
        for k in ret_vals.index.values:
            ret_dict[k] = [ret_vals[k]]
            ret_pct[k] = (data.iloc[period-1][k]-data.iloc[0][k])/(data.iloc[0][k]*period)
        return ret_dict, ret_pct
    else:
        smoothing = abs(smoothing)
        ret_pct = dict()
        start_p = -1
        for stock in data.columns:
            cur_comp = data.iloc[-1:][stock].values[0]*(smoothing/(1+period))
            
            if not (stock in init_v):
                init_v[stock] = [np.nanmean(data[stock].iloc[:-1])]
                
            past_comp = init_v[stock][start_p]*(1 - (smoothing/(1+period)))
                
            ret_pct[stock] = (cur_comp+past_comp-init_v[stock][start_p])/init_v[stock][start_p]
            init_v[stock].append(cur_comp+past_comp)
        return init_v, ret_pct

def adjust_cumulative(df, ar):
    for i in range(5, len(df.iloc[0])):
        df.iat[0, i] = np.nan
    l_cum, s_cum, cum = 1, 1, 1
    
    for r in range(1, len(df)):
        cum*=(1+df.iloc[r, 5])
        df.iat[r, 6]=(cum-1)
        #df.iat[r, 4]=cum*10000
        l_cum*= (1+df.iloc[r, 7])
        df.iat[r,8]=(l_cum-1)
        if ar:
            s_cum*= (1+df.iloc[r, 9])
            df.iat[r,10]=(s_cum-1)
    return df