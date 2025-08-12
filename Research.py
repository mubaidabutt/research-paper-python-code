import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import requests
from bs4 import BeautifulSoup
import feedparser
import urllib.parse
from collections import Counter
print("                    ---------------------- BASIC STOCK SCREENER on the basis of BENJAMIS METRICS----------------------")
print("                                                          Prepared by:")
print()
print("                                                        Hafsa Kamran LUMS")
print("                                                          Ali Ashar LUMS")
print("                                                     Muhammad Ubaida Butt LUMS")
print()
#Customized Criteria 
print("Enter your screening criteria OR press Enter to use default values from Benjamin's Book 'The Intelligent Investor'):")

try:
    pe_limit = float(input("Max PE ratio (Default 15): ") or 15)
    min_growth = float(input("Minimum EPS growth in percentage (Default 33): ") or 33)
    min_positive_years = int(input("Minimum years of positive earnings (Default 10): ") or 10)
    min_current_ratio = float(input("Minimum Current Ratio (Default 1.5):" or 1.5))
except ValueError:
    print("Invalid input. Using default values.")
    pe_limit = 15
    min_growth = 33
    min_positive_years = 10
    min_current_ratio = 1.5

#Load your Excel file
file_path = r"C:\Users\user\Downloads\KSE 100.xlsx"
df = pd.read_excel(file_path)

#Filter based on criteria
cond_pe = df['PE'] <= pe_limit

eps_growth_cols = ['EPS_Growth1', 'EPS_Growth2', 'EPS_Growth3', 'EPS_Growth4']
cond_growth = (df[eps_growth_cols] >= min_growth).all(axis=1)

earnings_columns = ['Earnings_Year1','Earnings_Year2','Earnings_Year3','Earnings_Year4']
cond_positive = (df[earnings_columns] > 0).sum(axis=1) >= min_positive_years

cond_current = df['CurrentRatio'] >= min_current_ratio
print()
#Getting Filtered Results
print('Following companies meet the given criteria:')
filtered_df = df[cond_pe & cond_growth & cond_positive & cond_current]
print(filtered_df)

# ----------------- SENTIMENT ANALYSIS SECTION -----------------
print("\nStarting Sentiment Analysis using FinBERT sentiment model...")
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ================== FUNCTIONS ==================
def get_news_headlines_rss(company_name, num_headlines=5):
    """Fetch latest news headlines from Google News RSS feed."""
    query = urllib.parse.quote(company_name)
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-PK&gl=PK&ceid=PK:en"
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries[:num_headlines]]

def analyze_sentiment(headlines):
    """Run FinBERT sentiment analysis and return majority label + score."""
    if not headlines:
        return "No data", 0
    results = sentiment_pipeline(headlines)
    counts = Counter([r['label'].lower() for r in results])
    total = sum(counts.values())
    sentiment = counts.most_common(1)[0][0].capitalize()
    sentiment_score = round((counts['positive'] - counts['negative']) / total, 3) if total > 0 else 0
    return sentiment, sentiment_score

# ================== SENTIMENT ANALYSIS ==================
print("\nRunning Sentiment Analysis on filtered companies...\n")

sentiments = []
scores = []

for company in filtered_df['COMPANY']:
    print(f"\n=== {company} ===")
    headlines = get_news_headlines_rss(company)
    if headlines:
        for idx, h in enumerate(headlines, start=1):
            print(f"{idx}. {h}")
    else:
        print("No headlines found.")

    sentiment, score = analyze_sentiment(headlines)
    print(f"Sentiment: {sentiment}, Score: {score}")

    sentiments.append(sentiment)
    scores.append(score)

# Assign columns using .loc to avoid SettingWithCopyWarning
filtered_df.loc[:, 'Sentiment'] = sentiments
filtered_df.loc[:, 'SentimentScore'] = scores

print("\nFinal sentiment analysis table:")
print(filtered_df[['COMPANY', 'Sentiment', 'SentimentScore']])