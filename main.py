# Import libraries
import streamlit as st
import requests
import yfinance as yf
import pandas as pd
from textblob import TextBlob

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

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    if text:  # Check if text is not None
        blob = TextBlob(text)
        return blob.sentiment.polarity  # Returns a score from -1 (negative) to 1 (positive)
    return 0  # Return neutral sentiment if text is None

# Streamlit App UI
st.title("Stock News and Sentiment Analysis")

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
        sentiment = analyze_sentiment(summary)
        published_date = article.get('publishedAt', '')  # Get the publication date
        news_summaries.append({
            "Title": title,
            "Summary": summary,
            "Sentiment": sentiment,
            "Date": published_date
        })

    # Create a DataFrame of news summaries and sentiment, sorted by date
    news_df = pd.DataFrame(news_summaries)
    news_df['Date'] = pd.to_datetime(news_df['Date'])  # Convert Date column to datetime
    news_df = news_df.sort_values(by='Date', ascending=False)  # Sort by date

    st.write("News and Sentiment Analysis", news_df)

    # Display sentiment analysis summary
    st.bar_chart(news_df["Sentiment"])
