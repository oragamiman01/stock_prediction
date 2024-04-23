"""
    This script obtains the raw data we need for per-ticker sentiment and places it in the specified data directory.

    Use the "StockDataWrapper.py" file to actually load the raw data from the data directory

    NOTE: need to create a credentials file and implement the methods imported below to return the API key strings

    Bottom of file has an explanation of how the obb framework works in python
"""

import os
from openbb import obb
from CREDENTIALS import get_pat, get_fmp_key, get_polygon_key, get_tiingo_key
import numpy as np
from newspaper import Article
import nltk
from transformers import pipeline


def save_raw_obb_article_urls(ticker, start_date, end_date, data_dir):
    obb.account.login(pat=get_pat())
    # https://docs.openbb.co/platform/usage/api_keys
    obb.user.credentials.fmp_api_key = get_fmp_key()
    obb.user.credentials.polygon_api_key = get_polygon_key()
    obb.user.credentials.tiingo_token = get_tiingo_key()
    obb.account.save()
    obb.account.refresh()

    obb_data = obb.news.company(symbol=ticker, start_date=start_date, end_date=end_date, limit=100000, provider="polygon").results
    arts = []
    for a in obb_data:
        dct = {
            "date": a.date,
            "title": a.title,
            "text": a.text,
            "url": a.url,
            "symbols": a.symbols
        }
        arts.append(dct)

    filepath = f'{data_dir}/sentimentsRaw/{ticker}'
    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    np.save(f'{filepath}/rawObbArticleData.npy', np.array(arts))
    print(f"saved raw article data for {ticker}")

    return arts


def save_sentiment_scores_for_ticker(ticker, data_dir):
    nltk.download('punkt')  # needed for Newspaper parsing

    arts = np.load(f"{data_dir}/sentimentsRaw/{ticker}/rawObbArticleData.npy", allow_pickle=True).tolist()
    date_scores = {}

    num_invalid_from_newspaper = 0
    num_invalid_from_finbert = 0

    for i, article in enumerate(arts):
        if i % 100 == 0:
            print(f"processed {i} articles")

        date = article["date"].strftime("%Y-%m-%d")
        url = article["url"]

        sentiment_score = _get_score_for_article(url)
        if sentiment_score[0] is None:
            if sentiment_score[1] == 1:
                num_invalid_from_newspaper += 1
            else:
                num_invalid_from_finbert += 1
            continue

        if date not in date_scores:
            date_scores[date] = []
        date_scores[date].append(sentiment_score[1])

    for k, v in date_scores.items():
        date_scores[k] = sum(v) / len(v)

    save_path = f"{data_dir}/sentimentsRaw/{ticker}"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    np.save(f"{save_path}/dailySentimentScores.npy", date_scores)
    print(f"Saved daily sentiment scores for {ticker}; Invalid from newspaper: {num_invalid_from_newspaper}; Invalid from finbert: {num_invalid_from_finbert}")


def _get_score_for_article(url):
    # https://github.com/codelucas/newspaper?tab=readme-ov-file
    text = ""
    summary = ""
    # some news sites are "smart" and prevent requests coming from scripts, will error out
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        text = article.text
        summary = article.summary
    except:
        return None, 1

    # summary and text are the possible useful things here - summary seems to be at least decent quality
    try:
        analyzer = pipeline("text-classification", model="ProsusAI/finbert")
        sentiment_label = analyzer(summary)[0]["label"]
        ret_label = 1 if sentiment_label == "positive" else 0 if sentiment_label == "neutral" else -1
        return True, ret_label
    except:
        return None, 2


# uncomment to actually execute the loading
# with polygon free tier, you have to do them one ticker at a time (a for loop invoking the api does not work, requests
# past the first one are rate limited and killed
#   Also max rate is only a couple per minute (even manually), and if it is exceeded it wont throw an error the
#   file size will just be very small (indicating nothing was downloaded)

#   ^ this only applies to the function "save_raw_obb_article_urls"

# Can also run these by importing the functions into a jupyter notebook

# ticker = "AAPL"
# start_date = "2014-06-02"
# end_date = "2024-01-31"
# data_dir = "../data"
#
# save_raw_obb_article_urls(ticker, start_date, end_date, data_dir)

# save_sentiment_scores_for_ticker(ticker, data_dir)




#####################################################################
# obb usage:

# General usage: https://docs.openbb.co/platform/usage/basic_syntax

# https://docs.openbb.co/platform/extensions/data_extensions

# obb has: [access thru obb.X ; see available with dir(obb.X)]
    # account (login)
    # coverage (obb.coverage.providers to get additional functionalities beyond core ; also obb.coverage.commands for better view)
        # this tells you which additional tools are available to modify all the below modules

    # crypto (https://docs.openbb.co/platform/reference/crypto)
    # currency (https://docs.openbb.co/platform/reference/currency)
    # derivatives (https://docs.openbb.co/platform/reference/derivatives)
    # economy (https://docs.openbb.co/platform/reference/economy)
    # !!!!!!!!!! equity (https://docs.openbb.co/platform/reference/equity) !!!!!!!!!!!!!!!
    # etf (https://docs.openbb.co/platform/reference/etf)
    # fixedincome (https://docs.openbb.co/platform/reference/fixedincome)
    # index (https://docs.openbb.co/platform/reference/index)
    # !!!!!!!!!!! news (https://docs.openbb.co/platform/reference/news) !!!!!!!!!!!!!!
    # reference (get info about current installation)
    # regulators (https://docs.openbb.co/platform/reference/regulators)
    # system (get current system info)
    # user (get user info)

    # not included in default installation: [https://docs.openbb.co/platform/extensions/toolkit_extensions]
        # econometrics (https://docs.openbb.co/platform/reference/econometrics)
        # quantitative (https://docs.openbb.co/platform/reference/quantitative)
        # technical (https://docs.openbb.co/platform/reference/technical)
        # commodity (https://docs.openbb.co/platform/reference/commodity)