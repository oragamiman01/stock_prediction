# Dataset (Prediction Goal) Variants
The dataset (input -> output correspondences) determines which type of model we will want to use.

At its core, the backing data used to generate the dataset is always a time series, where each token contains the observed values for a given day. We have multiple time series, one for each stock ticker.

Unless otherwise specified, each dataset (and the corresponding model to predict it) is only based on 1 ticker, which is given as input [i.e. one ML model per stock ticker].

In all cases, we want to predict future values in the series given prior values. The things we can vary in each dataset are:
- values used for each input token (e.g just open VS open, close, high, and low)
- number of input tokens used as the full input for each sample (1 week prior VS 1 year prior vs ...)
- the output domain (e.g. predicting the next day's close value VS predicting the next day's high VS just predicting either "higher" or "lower" than the previous day)
- how many days ahead we predict for (e.g. generate "higher" or "lower" for each of the next 5 days)
- do we want to do prediction based on (or for) multiple tickers? How would this modify the input? The output?

Here are some possible variations:

1. **Baseline** : A dataset where each token just contains 1 value corresponding to the closing stock price per day. The input will be 7 tokens long, and the output will be the regressed next day's close value. Predicts 1 day ahead (encoder-only network).
2. **All_Knobs** : A dataset where each token contains all values available for that day. The input will be 365 tokens long, and the output will be the regressed next day's close value. Predicts 1 day ahead (encoder-only network)
3. **Future** : Same as All_Knobs, but predicts each of the 5 next days (encoder-decoder network)

For any of the given datasets (X), any combination of these variations can be added:
1. **Binary** : Same as X, but predicts higher or lower close price relative to the previous day
2. **Long_history** : Same as X, but uses all available prior tokens as input in order to predict the given day (variable length input)
3. **Fully_informed** : Same as X, but the input tokens include all information from ALL stock tickers from that day, stacked and flattened into one vector. The goal is still to predict the target/output for just ONE stock ticker [1 model per ticker]

Here are the values we have available for each day, for a given ticker:
- open, close, high, low prices
- some obb technical processed data (described in StockDataWrapper.py)
- the absolute date (mm/dd/yyyy)
  - will this cause an issue because weekends are skipped?
- pre-made sentiment metrics that are within the range [-1,1] (daily sentiment index)
- per-ticker daily sentiment metrics in the range [-1, 1] that we extracted [by scraping historical news articles and passing them through FinBert]
  - we only have this for AAPL, MSFT, TSLA, NVDA, and AMZN, and only for ~365 distinct dates (we need to pay to get more data)
- If working with multiple tickers in one model, we have the ticker ID and the sector of the company

In the transformer, we have the options of:
- using a learned embedding vs just using the actual input vector as the embedding vector and moving on to the positional encoding step immediately
- ?


## Things to consider
- Should we try a variant where we limit the dataset to just dates where we have per-ticker sentiment values (>= 2021?)