# Import libraries
import streamlit as st
import requests
import yfinance as yf
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

# Load FinBERT model and tokenizer
@st.cache_resource  # Caches the model and tokenizer to avoid reloading
def load_finbert():
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_finbert()

# Function to analyze sentiment using FinBERT
def analyze_sentiment_finbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = predictions[0].detach().numpy()
    sentiment = ["negative", "neutral", "positive"]
    return sentiment[sentiment_score.argmax()], sentiment_score.max()

# Function to get stock data using Yahoo Finance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="1mo")

# Function to fetch news data using News API
def get_news(api_key, query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

# Streamlit App UI
st.title("Financial News and Sentiment Analysis")

# Inputs for API key and stock ticker
api_key = st.text_input("Enter your News API Key")
ticker = st.text_input("Enter Stock Ticker")

# Run analysis when button is pressed
if st.button("Analyze"):
    # Fetch and display stock data
    stock_data = get_stock_data(ticker)
    st.write(f"Stock Price Data for {ticker}")
    st.line_chart(stock_data['Close'])

    # Fetch and display news articles
    news_articles = get_news(api_key, ticker)
    news_summaries = []
    for article in news_articles:
        title = article['title']
        summary = article.get('description', '')  # Use empty string if description is None
        sentiment_label, sentiment_score = analyze_sentiment_finbert(summary)
        published_date = article.get('publishedAt', '')  # Get the publication date
        news_summaries.append({
            "Title": title,
            "Summary": summary,
            "Sentiment": sentiment_label,
            "Sentiment Score": sentiment_score,
            "Date": published_date
        })

    # Create a DataFrame of news summaries and sentiment, sorted by date
    news_df = pd.DataFrame(news_summaries)
    news_df['Date'] = pd.to_datetime(news_df['Date'])  # Convert Date column to datetime
    news_df = news_df.sort_values(by='Date', ascending=False)  # Sort by date

    st.write("News and Sentiment Analysis", news_df)

    # Display sentiment analysis summary
    sentiment_counts = news_df['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
