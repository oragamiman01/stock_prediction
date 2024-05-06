from datetime import datetime

from StockDataWrapper import get_time_series
from torch.utils.data import Dataset
import torch


## Implementation of the "Baseline" dataset described in "Process.md" ##
# note that the label consists of 2 values - if X is the day we want to predict, the label has the close price for X and
# the close price for day (X-1) - this is because of the way the loss is computed in Models.py (the Agent class)
class BaselineDataset(Dataset):
    def __init__(self, ticker, data_dir):
        # Is a list of dictionaries. Contains values described at top of StockDataWrapper.py
        ordered_time_series = get_time_series(ticker, data_dir)
        self.X = []
        self.y = []

        # need to get 7 days of input tokens and predict the 8th day, and so on
        for cur_target_idx in range(7, len(ordered_time_series)):

            cur_target_actual_after = ordered_time_series[cur_target_idx]["close"]
            cur_target_actual_prev_day = ordered_time_series[cur_target_idx - 1]["close"]
            cur_input = []

            for i in range(cur_target_idx - 7, cur_target_idx):
                close = ordered_time_series[i]["close"]
                avg = close
                cur_input.append([avg])

            self.X.append(cur_input)
            self.y.append([cur_target_actual_after, cur_target_actual_prev_day])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


## Implementation of the "All_Knobs" dataset described in "Process.md" ##
# note that the label consists of 2 values - if X is the day we want to predict, the label has the close price for X and
# the close price for day (X-1) - this is because of the way the loss is computed in Models.py (the Agent class)
class AllKnobsDataset_train(Dataset):
    def __init__(self, ticker, data_dir):
        # Is a list of dictionaries. Contains values described at top of StockDataWrapper.py
        ordered_time_series = get_time_series(ticker, data_dir)
        self.X = []
        self.y = []

        # get the unix timestamp value for 2024-04-15. Used to scale the date value used as input to the model
        cur_unix_timestamp = datetime(year=2024, month=4, day=15).timestamp()

        # need to get 365 days of input tokens and predict the 366th day, and so on
        for cur_target_idx in range(365, len(ordered_time_series) - 365):

            cur_target_actual_after = ordered_time_series[cur_target_idx]["close"]
            cur_target_actual_prev_day = ordered_time_series[cur_target_idx - 1]["close"]
            # each "sample" (input) will have multiple tokens. Each token is a vector of the day's values
            cur_input = []

            for i in range(cur_target_idx - 365, cur_target_idx):
                date = ordered_time_series[i]["date"]
                date = (datetime.strptime(date, "%Y-%m-%d").timestamp()) / cur_unix_timestamp  # scale to small value
                open = ordered_time_series[i]["open"]
                close = ordered_time_series[i]["close"]
                high = ordered_time_series[i]["high"]
                low = ordered_time_series[i]["low"]
                volume = ordered_time_series[i]["volume"]
                pct_change = ordered_time_series[i]["pct_change"]
                close_RSI_14 = ordered_time_series[i]["close_RSI_14"]
                ADOSC_3_10 = ordered_time_series[i]["ADOSC_3_10"]
                AROOND_25 = ordered_time_series[i]["AROOND_25"]
                AROONU_25 = ordered_time_series[i]["AROONU_25"]
                AROONOSC_25 = ordered_time_series[i]["AROONOSC_25"]
                CCI_14_0_015 = ordered_time_series[i]["CCI_14_0.015"]
                CG_14 = ordered_time_series[i]["CG_14"]
                close_HMA_50 = ordered_time_series[i]["close_HMA_50"]
                ISA_9 = ordered_time_series[i]["ISA_9"]
                ISB_26 = ordered_time_series[i]["ISB_26"]
                ITS_9 = ordered_time_series[i]["ITS_9"]
                IKS_26 = ordered_time_series[i]["IKS_26"]
                VWAP_D = ordered_time_series[i]["VWAP_D"]

                sent_general = ordered_time_series[i]["sentiment_general_news"]
                sent_specific = ordered_time_series[i]["ticker_specific_sentiment"]

                cur_token = [date, open, close, high, low, volume, pct_change, close_RSI_14, ADOSC_3_10, AROOND_25,
                             AROONU_25, AROONOSC_25, CCI_14_0_015, CG_14, close_HMA_50, ISA_9, ISB_26, ITS_9,
                             IKS_26, VWAP_D, sent_general, sent_specific]

                cur_input.append(cur_token)

            self.X.append(cur_input)
            self.y.append([cur_target_actual_after, cur_target_actual_prev_day])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class AllKnobsDataset_test(Dataset):
    def __init__(self, ticker, data_dir):
        # Is a list of dictionaries. Contains values described at top of StockDataWrapper.py
        ordered_time_series = get_time_series(ticker, data_dir)
        self.X = []
        self.y = []

        # get the unix timestamp value for 2024-04-15. Used to scale the date value used as input to the model
        cur_unix_timestamp = datetime(year=2024, month=4, day=15).timestamp()

        # need to get 365 days of input tokens and predict the 366th day, and so on
        for cur_target_idx in range(len(ordered_time_series) - 365, len(ordered_time_series)):

            cur_target_actual_after = ordered_time_series[cur_target_idx]["close"]
            cur_target_actual_prev_day = ordered_time_series[cur_target_idx - 1]["close"]
            # each "sample" (input) will have multiple tokens. Each token is a vector of the day's values
            cur_input = []

            for i in range(cur_target_idx - 365, cur_target_idx):
                date = ordered_time_series[i]["date"]
                date = (datetime.strptime(date, "%Y-%m-%d").timestamp()) / cur_unix_timestamp  # scale to small value
                open = ordered_time_series[i]["open"]
                close = ordered_time_series[i]["close"]
                high = ordered_time_series[i]["high"]
                low = ordered_time_series[i]["low"]
                volume = ordered_time_series[i]["volume"]
                pct_change = ordered_time_series[i]["pct_change"]
                close_RSI_14 = ordered_time_series[i]["close_RSI_14"]
                ADOSC_3_10 = ordered_time_series[i]["ADOSC_3_10"]
                AROOND_25 = ordered_time_series[i]["AROOND_25"]
                AROONU_25 = ordered_time_series[i]["AROONU_25"]
                AROONOSC_25 = ordered_time_series[i]["AROONOSC_25"]
                CCI_14_0_015 = ordered_time_series[i]["CCI_14_0.015"]
                CG_14 = ordered_time_series[i]["CG_14"]
                close_HMA_50 = ordered_time_series[i]["close_HMA_50"]
                ISA_9 = ordered_time_series[i]["ISA_9"]
                ISB_26 = ordered_time_series[i]["ISB_26"]
                ITS_9 = ordered_time_series[i]["ITS_9"]
                IKS_26 = ordered_time_series[i]["IKS_26"]
                VWAP_D = ordered_time_series[i]["VWAP_D"]

                sent_general = ordered_time_series[i]["sentiment_general_news"]
                sent_specific = ordered_time_series[i]["ticker_specific_sentiment"]

                cur_token = [date, open, close, high, low, volume, pct_change, close_RSI_14, ADOSC_3_10, AROOND_25,
                             AROONU_25, AROONOSC_25, CCI_14_0_015, CG_14, close_HMA_50, ISA_9, ISB_26, ITS_9,
                             IKS_26, VWAP_D, sent_general, sent_specific]

                cur_input.append(cur_token)

            self.X.append(cur_input)
            self.y.append([cur_target_actual_after, cur_target_actual_prev_day])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class VariableLengthDataset(Dataset):
    def __init__(self, tickers: list[str], data_dir: str, lag_days: int, test_days: int, test: bool):
        self.X = []
        self.y = []
        self.tickers = []

        num_to_ticker = dict()
        ticker_to_num = dict()
        ids = np.linspace(-1, 1, len(tickers)).round(5)
        for i, num in enumerate(ids):
            num_to_ticker[str(num)] = tickers[i]
            ticker_to_num[tickers[i]] = num

        # get the unix timestamp value for 2024-04-15. Used to scale the date value used as input to the model
        cur_unix_timestamp = datetime.datetime(year=2024, month=4, day=15).timestamp()

        for ticker in tickers:
            ordered_time_series = get_time_series(ticker, data_dir, normalize=True)

            if test:
                idx = range(len(ordered_time_series) - lag_days - test_days, len(ordered_time_series))
            else:
                idx = range(lag_days, len(ordered_time_series) - lag_days - test_days)

            # need to get lag_days days of input tokens and predict the lag_days+1 day, and so on
            for cur_target_idx in idx:

                # get target values
                cur_target_actual_after = [ticker_to_num[ticker]]
                for key in ordered_time_series[cur_target_idx]:

                    val = ordered_time_series[cur_target_idx][key]
                    if key == 'date':
                        val = (datetime.datetime.strptime(val, "%Y-%m-%d").timestamp()) / cur_unix_timestamp  # scale to small value
                    cur_target_actual_after.append(val)

                # each "sample" (input) will have multiple tokens. Each token is a vector of the day's values
                cur_input = []

                for i in range(cur_target_idx - lag_days, cur_target_idx):

                    cur_token = [ticker_to_num[ticker]]
                    for key in ordered_time_series[cur_target_idx]:

                        val = ordered_time_series[cur_target_idx][key]
                        if key == 'date':
                            val = (datetime.datetime.strptime(val, "%Y-%m-%d").timestamp()) / cur_unix_timestamp  # scale to small value
                        cur_token.append(val)

                    cur_input.append(cur_token)

                self.X.append(cur_input)
                self.y.append(cur_target_actual_after)
                self.tickers.append(ticker)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), self.tickers[idx]
    

class IncrementalDataset(Dataset):
    def __init__(self, tickers: list[str], data_dir: str, lag_days: int, test_days: int, test: bool):
        self.X = []
        self.y = []
        self.tickers = []

        num_to_ticker = dict()
        ticker_to_num = dict()
        ids = np.linspace(-1, 1, len(tickers)).round(5)
        for i, num in enumerate(ids):
            num_to_ticker[str(num)] = tickers[i]
            ticker_to_num[tickers[i]] = num

        # get the unix timestamp value for 2024-04-15. Used to scale the date value used as input to the model
        cur_unix_timestamp = datetime.datetime(year=2024, month=4, day=15).timestamp()

        for ticker in tickers:
            ordered_time_series = get_time_series(ticker, data_dir, normalize=True)

            if test:
                idx = range(len(ordered_time_series) - lag_days - test_days, len(ordered_time_series) - lag_days - test_days + 1)
            else:
                idx = range(lag_days + 1500, len(ordered_time_series) - lag_days - test_days) # CUT OFF OLD PART OF DATA!!!!!!!!!!!!! FYI

            # need to get lag_days days of input tokens and predict the lag_days+1 day, and so on
            for cur_target_idx in idx:

                # get target values
                cur_target_actual_after = [ticker_to_num[ticker]]
                for key in ordered_time_series[cur_target_idx]:

                    val = ordered_time_series[cur_target_idx][key]
                    if key == 'date':
                        val = (datetime.datetime.strptime(val, "%Y-%m-%d").timestamp()) / cur_unix_timestamp  # scale to small value
                    cur_target_actual_after.append(val)

                # each "sample" (input) will have multiple tokens. Each token is a vector of the day's values
                cur_input = []

                for i in range(cur_target_idx - lag_days, cur_target_idx):

                    cur_token = [ticker_to_num[ticker]]
                    for key in ordered_time_series[cur_target_idx]:

                        val = ordered_time_series[cur_target_idx][key]
                        if key == 'date':
                            val = (datetime.datetime.strptime(val, "%Y-%m-%d").timestamp()) / cur_unix_timestamp  # scale to small value
                        cur_token.append(val)

                    cur_input.append(cur_token)

                self.X.append(cur_input)
                self.y.append(cur_target_actual_after)
                self.tickers.append(ticker)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), self.tickers[idx]

