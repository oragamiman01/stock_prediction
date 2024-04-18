import numpy as np
# import torch
import pandas as pd


def directional_accuracy(gt, pred):
    """Calculates directional accuracy of given ground truth and prediction series.
    From Kaeley et al.
    
    inputs:
        gt: ground truth prices
        pred: predicted prices
        
    returns:
        acc: directional accuracy of predicted values
    """
    acc = []
    for i in range(1, len(gt)):
        if gt[i] >= gt[i-1] and pred[i] >= gt[i-1]:
            acc.append(1)
        elif gt[i] < gt[i-1] and pred[i] < gt[i-1]:
            acc.append(1)
        else:
            acc.append(0)

    return np.array(acc).mean()

def directional_accuracy_pct_change(gt_pct, pred_pct):
    """Calculates directional accuracy of given ground truth and 
    prediction percent change series.
    From Kaeley et al.
    
    inputs:
        gt_pct: ground truth pct_change
        pred_pct: predicted pct_change
        
    returns:
        acc: directional accuracy of predicted values
    """
    gt_sign = np.sign(np.asarray(gt_pct))
    pred_sign = np.sign(np.asarray(pred_pct))
    return np.sum(gt_sign == pred_sign) / len(gt_pct)

def directional_accuracy_prob(gt_pct, pred_pct):
    """Calculates directional accuracy of given ground truth and 
    prediction percent change series.
    From Kaeley et al.
    
    inputs:
        gt_pct: ground truth pct_change
        pred_pct: predicted probability of positive pct_change
        
    returns:
        acc: directional accuracy of predicted values
    """
    gt_sign = np.sign(np.asarray(gt_pct))
    pred_sign = np.array(pred_pct)
    pred_sign = pred_sign >= 0.5
    return np.sum(gt_sign == pred_sign) / len(gt_pct)

def rename_cols(df, suffix, exclude):
    new_names = dict()
    cols = list(df.columns)
    for n in exclude:
        cols.remove(n)
    for c in cols:
        new_names[c] = c + suffix

    df = df.rename(columns=new_names)
    return df


def z_norm(df, col_exclude: list = None):
    """Performs z-score normalization on all columns of df except col_exclude
    
    inputs:
        df: stock data
        col_exclude: columns to be excluded from normalization
        
    returns:
        df_std: normalized z-score data
        stat_dict: dict where keys are each normalized column name,
                   values are tuple of (mean, std)
    """

    stat_dict = dict()
    df_std = df.copy()
    cols = list(df.columns)
    [cols.remove(c) for c in col_exclude]
    for c in cols:
        stat_dict[c] = (df[c].mean(), df[c].std()) #
        df_std[c] = (df[c] - df[c].mean()) / df[c].std()

    return df_std, stat_dict


def reverse_z_norm(df, stat_dict, col_exclude=None):
    """Reverses z-score normalization on all columns except col_exclude
    
    inputs:
        df: normalized stock data
        stat_dict: dict returned by z_norm
        col_exclude: columns to be excluded
        
    returns:
        df_real: dataframe of unnormalized columns
    """

    df_real = df.copy()
    cols = list(df.columns)
    [cols.remove(c) for c in col_exclude]
    for c in cols:
        mean, std = stat_dict[c]
        df_real[c] = df[c] * std + mean # reverse z score calculation to restore original data

    return df_real


def to_sequences(seq_size: int, obs: np.array):
    """Splits a table of data into sequences of given length"""

    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = obs[i:(i + seq_size), :]
        after_window = obs[i + seq_size, :]
        x.append(window)
        y.append(after_window)
    return x, y

def process_results(model_out, batch_size: int, 
                    batch_num: int, dates: list, tickers: list):
    """Function to process output tensor from network into usable data.
    Each batch output needs to be put in the correct list for its company

    inputs:
        model_out: tensor of size (batch_size, num_features)
        batch_size: batch size of dataloader
        batch_num: which batch is the loop on
        dates: list of dates that line up with the rows of the output data
        tickers: list of ticker symbols

    returns:
        processed: list of arrays
    """

    processed = []

    for idx, batch_out in enumerate(model_out):
        date_idx = batch_num * batch_size + idx
        date = dates[date_idx] # find date of predicted day
        
        pred_list = batch_out.detach().tolist()
        # ticker is last column since date was dropped
        # change from model output ticker to character ticker
        pred_list.append(tickers[idx])
        pred_list.append(date) # add prediction date to list
        processed.append(pred_list) # add to overall 2d array
        
    return processed

def process_results_incremental(model_out, date: pd.Timestamp, tickers: list):
    """Function to process output tensor from network into usable data.
    Each batch output needs to be put in the correct list for its company

    inputs:
        model_out: tensor of size (batch_size, num_features)
        batch_size: batch size of dataloader
        batch_num: which batch is the loop on
        dates: list of dates that line up with the rows of the output data
        tickers: list of ticker symbols

    returns:
        processed: list of arrays
    """

    processed = []

    for idx, batch_out in enumerate(model_out):
        pred_list = batch_out.detach().tolist()
        # ticker is last column since date was dropped
        # change from model output ticker to character ticker
        pred_list.append(tickers[idx])
        pred_list.append(date[tickers[idx]]) # add prediction date to list
        processed.append(pred_list) # add to overall 2d array
        
    return processed