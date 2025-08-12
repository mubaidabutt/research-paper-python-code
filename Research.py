import pandas as pd
import urllib.parse
import feedparser
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# My Functions
def load_data(path):
    return pd.read_excel(path)

def filter_stocks(df, pe, growth, pos_years, current_ratio):
    cond_pe = df['PE'] <= pe
    cond_growth = (df[['EPS_Growth1', 'EPS_Growth2', 'EPS_Growth3', 'EPS_Growth4']] >= growth).all(axis=1)
    cond_positive = (df[['Earnings_Year1','Earnings_Year2','Earnings_Year3','Earnings_Year4']] > 0).sum(axis=1) >= pos_years
    cond_current = df['CurrentRatio'] >= current_ratio
    return df[cond_pe & cond_growth & cond_positive & cond_current]

def get_news_headlines(company, n=5):
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(company)}&hl=en-PK&gl=PK&ceid=PK:en"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:n]]

def analyze_sentiment(headlines, pipeline):
    if not headlines:
        return "No data", 0
    results = pipeline(headlines)
    counts = Counter(r['label'].lower() for r in results)
    total = sum(counts.values())
    sentiment = counts.most_common(1)[0][0].capitalize()
    score = round((counts.get('positive',0) - counts.get('negative',0)) / total, 3) if total else 0
    return sentiment, score

# Main Code
print("                                          ------ BASIC STOCK SCREENER based on BENJAMIN'S METRICS ------")
print("Prepared by:\nHafsa Kamran LUMS\nAli Ashar LUMS\n Muhammad Ubaida Butt LUMS\n")

pe_limit = float(input("Max PE ratio (Default 15): ") or 15)
min_growth = float(input("Minimum EPS growth in percentage (Default 33): ") or 33)
min_positive_years = int(input("Minimum no. of years with positive earnings (Default 10): ") or 10)
min_current_ratio = float(input("Min Current Ratio (Default 1.5): ") or 1.5)

df = load_data(r"C:\Users\user\Downloads\KSE 100.xlsx")
filtered_df = filter_stocks(df, pe_limit, min_growth, min_positive_years, min_current_ratio)

print("\nFollowing companies meet the criteria:")
print(filtered_df)

# Sentiment Analysis
print("\nStarting sentiment analysis using FinBERT Sentiment Analysis Model")
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiments, scores = [], []

for company in filtered_df['COMPANY']:
    print(f"\n=== {company} ===")
    headlines = get_news_headlines(company)
    if headlines:
        for i, h in enumerate(headlines, 1):
            print(f"{i}. {h}")
    else:
        print("No headlines found.")

    sentiment, score = analyze_sentiment(headlines, sentiment_pipe)
    print(f"Sentiment: {sentiment}, Score: {score}")
    sentiments.append(sentiment)
    scores.append(score)

filtered_df.loc[:, 'Sentiment'] = sentiments
filtered_df.loc[:, 'SentimentScore'] = scores

print("\nFinal sentiment analysis table:")
print(filtered_df[['COMPANY', 'Sentiment', 'SentimentScore']])
