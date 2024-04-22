from StockDataWrapper import get_time_series
from torch.utils.data import Dataset
import torch


## Implementation of the "Baseline" dataset described in "Process.md" ##
class BaselineDataset(Dataset):
    def __init__(self, ticker, data_dir):
        # Is a list of dictionaries. Contains values described at top of StockDataWrapper.py
        ordered_time_series = get_time_series(ticker, data_dir)
        self.X = []
        self.y = []

        # need to get 7 days of input tokens and predict the 8th day, and so on
        for cur_target_idx in range(7, len(ordered_time_series)):

            cur_target = ordered_time_series[cur_target_idx]["close"]
            cur_input = []

            for i in range(cur_target_idx - 7, cur_target_idx):
                close = ordered_time_series[i]["close"]
                avg = close
                cur_input.append(avg)

            self.X.append(cur_input)
            self.y.append(cur_target)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
