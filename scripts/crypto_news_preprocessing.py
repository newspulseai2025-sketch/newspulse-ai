import pandas as pd
import re
from datetime import datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not installed
nltk.download('vader_lexicon')


# 1. LOAD NEWS DATA
file_path = r"text\crypto_project\data\raw\investing_news_crypto_data.csv"
news_df = pd.read_csv(file_path)

print("Original columns:", news_df.columns)


# 2. CLEAN TEXT
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+", "", text)          # remove links
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)  # remove special chars
    text = re.sub(r"\s+", " ", text).strip()     # remove extra spaces
    return text

news_df["title_clean"] = news_df["title"].astype(str).apply(clean_text)
news_df["description_clean"] = news_df["description"].astype(str).apply(clean_text)


# 3. PARSE DATE
news_df["date"] = pd.to_datetime(news_df["date"], errors='coerce')
news_df = news_df.dropna(subset=["date"])  # drop rows where date parsing failed


# 4. SENTIMENT SCORE (USING VADER)
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return sia.polarity_scores(str(text))["compound"]

news_df["sentiment"] = news_df["description_clean"].apply(get_sentiment)


# 5. AGGREGATE NEWS BY DATE
daily_news = news_df.groupby(news_df["date"].dt.date).agg(
    news_count=("title", "count"),
    mean_sentiment=("sentiment", "mean")
).reset_index()

daily_news.rename(columns={"date": "date"}, inplace=True)
daily_news["date"] = pd.to_datetime(daily_news["date"])

print("Daily news summary created!")
print(daily_news.head())

# SAVE RESULT (OPTIONAL)
daily_news.to_csv("text\crypto_project\data\processed\daily_news_aggregated.csv", index=False)
