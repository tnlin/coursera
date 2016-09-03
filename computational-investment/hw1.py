# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
def print_result(startdate, enddate, ls_symbols, portfolio, sharpe,vol, daily_ret, cum_ret):
    print "Start Date:",startdate
    print "End Date:",enddate
    print "Symbols:",ls_symbols
    print "Optimal Allocations:", portfolio
    print "-----------------------------------------"
    print "Sharpe Ratio:", sharpe
    print "Volatility (stdev of daily returns):", vol
    print "Average Daily Return:", daily_ret
    print "Cumulative Return:", cum_ret

def main():
    # Start and End date of the charts
    #startdate = dt.datetime(2011, 1, 1)
    #enddate = dt.datetime(2011, 12, 31)

    # List of symbols
    #ls_symbols = ['AAPL','GLD','GOOG','XOM']

    #portfolio = [0.4, 0.4, 0.0, 0.2]
    #vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, ls_symbols, portfolio)
    #print_result(startdate, enddate, ls_symbols, portfolio, sharpe,vol, daily_ret, cum_ret)


    # Optimize portfolio
    startdate = dt.datetime(2011, 1, 1)
    enddate = dt.datetime(2011, 12, 31)
    ls_symbols = ['AAPL','GOOG','IBM','MSFT']

    sharpe, portfolio = optimize_portfolio(startdate, enddate, ls_symbols)
    print "Best Sharpe Ratio:", sharpe
    print "Best Portfolio:", portfolio

def optimize_portfolio(startdate, enddate, ls_symbols):
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo', cachestalltime=0)

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    df_rets = d_data['close'].copy()
    df_rets = df_rets.fillna(method='ffill')
    df_rets = df_rets.fillna(method='bfill')
    df_rets = df_rets.fillna(1.0)
    df_rets_val = df_rets.values

    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = df_rets_val / df_rets_val[0]

    idx = np.arange(-1, 1, 0.1)
    best_portfolio = []
    best_sharpe = 0
    for i in idx:
        print "Start: i=",i
        for j in idx:
            for k in idx:
                if i+j+k > 2 or i+j+k < 0:
                    continue
                else:
                    # Weighted by portfolio
                    portfolio = [i,j,k, 1-i-j-k]
                    port_val = np.dot(na_normalized_price, portfolio)

                    # returnize0 works on ndarray and not dataframes.
                    tsu.returnize0(port_val)

                    # Return Values
                    daily_ret = np.mean(port_val)
                    vol       = np.std(port_val)
                    sharpe    = math.sqrt(252) * daily_ret / vol
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_portfolio = portfolio
    '''
    # Plotting the plot of daily returns
    port_val = np.dot(na_normalized_price, best_portfolio)
    plt.clf()
    plt.plot(ldt_timestamps, port_val)
    plt.plot(ldt_timestamps, na_normalized_price[:,3])
    plt.axhline(y=0, color='r')
    plt.legend(['Best', '$SPX'])
    plt.ylabel('Daily Returns')
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.savefig('best.pdf', format='pdf')
    '''

    return best_sharpe, [round(i,2) for i in best_portfolio]


def simulate(startdate, enddate, ls_symbols, portfolio):
    ''' Main Function'''
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo', cachestalltime=0)

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    df_rets = d_data['close'].copy()
    df_rets = df_rets.fillna(method='ffill')
    df_rets = df_rets.fillna(method='bfill')
    df_rets = df_rets.fillna(1.0)
    df_rets_val = df_rets.values

    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = df_rets_val / df_rets_val[0]

    # Weighted by portfolio
    port_val = np.dot(na_normalized_price, portfolio)

    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(port_val)

    # Return Values
    daily_ret = np.mean(port_val)
    vol       = np.std(port_val)
    sharpe    = math.sqrt(252) * daily_ret / vol
    cum_ret   = np.cumprod(port_val+1)[-1]

    return vol, daily_ret, sharpe, cum_ret

if __name__ == '__main__':
    main()
