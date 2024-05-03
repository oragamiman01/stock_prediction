"""
Helper functions to wrap our raw data in an easy-to-use-and-extract format for making custom PyTorch Datasets

Assuming all relevant raw data is in ../data/obbRaw and ../data/sentimentsRaw

Each returned time series (list) has entries that are dictionaries. Each dictionary has keys as incicated in square
brackets [] below

This file is meant to be the "messy" or "unstructured" one so that all other files can use a more standard format
    - aggregating all the different raw data formats into one unit

For raw stock numbers (under obbRaw), we currently have a single time series for each ticker as a CSV file. Each
observation (day) contains:
    - Absolute Date (YYYY-MM-DD) [date]
    - Open price [open]
    - Close price [close]
    - High price [high]
    - Low price [low]
    - Volume [volume]
    - Percent Change [pct_change]
    - [close_RSI_14]
    - [ADOSC_3_10]
    - [AROOND_25]
    - [AROONU_25]
    - [AROONOSC_25]
    - [CCI_14_0.015]
    - [CG_14]
    - [close_HMA_50]
    - [ISA_9]
    - [ISB_26]
    - [ITS_9]
    - [IKS_26]
    - [VWAP_D]

For Sentiment metrics, we have the following:

    - [sentiment_general_news] 'general_news_sentiment_data.csv' : a file that contains a single number in the range
      [-1, 1] for each day from 1980 to April 2024
        - https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/
        - NOTE: this one includes weekends - could be useful to somehow aggregate the weekends into either friday
          or monday, but they are ignored as of now

    - [ticker_specific_sentiment] per-ticker daily sentiment metric : this information is available under
      ../data/sentimentsRaw/tickers/*. It can be obtained by running the "SentimentPerTickerObtainer" file.
        - We only have this for AAPL, MSFT, TSLA, NVDA, and AMZN
        - and only for ~365 distinct dates (we need to pay to get more data)

"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def get_ticker_list(data_dir):
    """
        Generate a numpy array of all tickers in {data_dir}/constituents.csv
    """

    df = pd.read_csv(f'{data_dir}/constituents.csv')
    # first col is ticker, second is company name, third is sector
    column_headers = df.columns.tolist()

    ticker_series = df[column_headers[0]]
    ticker_array = ticker_series.to_numpy()
    ticker_array = np.array([ticker for ticker in ticker_array if ticker != "CVS"])

    return ticker_array


def get_time_series(ticker, data_dir, normalize=False):
    """
        Generate a time series (list) of dictionaries for the given ticker. The dictionaries contain values described
        at the top of this file
    """

    # the raw stock data contains, for each day, the entries in the javadoc comments at the top of the file
    # initialize our time series with the raw OBB data - kept in order of date
    time_series = _helper_load_obb_raw_historical(ticker, data_dir, normalize)

    # add the general_news_sentiment data, ensuring the dates align
    time_series = _helper_add_general_news_sentiment(time_series, data_dir)

    # add the ticker-specific daily sentiment metric
    time_series = _helper_add_ticker_specific_sentiment_if_exists(time_series, ticker, data_dir)

    return time_series


def get_all_time_series(data_dir, normalize=False):
    """
        Generate a dictionary of {ticker -> time series} entries for all tickers in our data bank.
        Each entry in a given time series contains the values described at the top of this file
    """

    all_tickers = get_ticker_list(data_dir)
    return {ticker: get_time_series(ticker, data_dir, normalize) for ticker in all_tickers}

def get_some_time_series(tickers: list, data_dir, normalize=False):
    """
        Same as get_all_dime_series except for a specific list of company tickers.
    """

    return {ticker: get_time_series(ticker, data_dir, normalize) for ticker in tickers}


############################################################################################################

def _helper_load_obb_raw_historical(ticker, data_dir, z_norm=False):
    if not os.path.isdir(f'{data_dir}/obbRaw/{ticker}'):
        raise Exception(f"There is no directory at '{data_dir}/obbRaw/{ticker}'")

    csv_files = [f for f in os.listdir(f'{data_dir}/obbRaw/{ticker}') if f.endswith('.csv')]
    if len(csv_files) != 1:
        raise Exception(f"Incorrect format for directory '{data_dir}/obbRaw/{ticker}' - is there a single csv file?")

    time_series_file = os.path.join(f'{data_dir}/obbRaw/{ticker}', csv_files[0])
    time_series_df = pd.read_csv(time_series_file)
    time_series_df = time_series_df.iloc[:, 1:]
    if z_norm == True: # normalize data if desired
        time_series_df = _helper_z_norm(time_series_df, col_exclude=['date', 'pct_change'])
    raw_stock_data_series = time_series_df.to_dict(orient='records')

    return raw_stock_data_series

def _helper_z_norm(df, col_exclude: list = None):
    """Performs z-score normalization on all columns of df except col_exclude
    
    inputs:
        df: raw stock data
        col_exclude: columns to be excluded from normalization
        
    returns:
        df_std: zero-mean unit-variance standardized data
    """

    df_std = df.copy()
    cols = list(df.columns)
    [cols.remove(c) for c in col_exclude]
    for c in cols:
        df_std[c] = (df[c] - df[c].mean()) / df[c].std()

    return df_std


def _helper_add_general_news_sentiment(time_series, data_dir):
    sentiment_general_df = pd.read_csv(f"{data_dir}/sentimentsRaw/general_news_sentiment_data.csv")
    sentiment_general_series = sentiment_general_df.to_dict(orient='records')
    cur_idx = 0
    for observation in time_series:
        date_to_match = observation["date"]  # in YYYY-MM-DD format
        date_converter = lambda dt: datetime.strptime(dt, "%m/%d/%y").strftime("%Y-%m-%d")

        # not an O(n^2) computation - just meant to align cur_idx
        while date_to_match != date_converter(sentiment_general_series[cur_idx]["date"]):
            cur_idx += 1

        assert date_to_match == date_converter(sentiment_general_series[cur_idx]["date"])  # just bc

        observation["sentiment_general_news"] = sentiment_general_series[cur_idx]["News Sentiment"]
        cur_idx += 1

    return time_series


def _helper_add_ticker_specific_sentiment_if_exists(time_series, ticker, data_dir):
    if not os.path.isdir(f'{data_dir}/sentimentsRaw/{ticker}'):
        map_dates_to_numerical_score = {}
    else:
        map_dates_to_numerical_score = np.load(f"{data_dir}/sentimentsRaw/{ticker}/dailySentimentScores.npy", allow_pickle=True).item()

    for observation in time_series:
        date = observation["date"]
        cur_num = map_dates_to_numerical_score[date] if date in map_dates_to_numerical_score else 0
        observation["ticker_specific_sentiment"] = cur_num

    return time_series
