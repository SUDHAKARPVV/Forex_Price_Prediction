import feedparser
import pandas as pd
from datetime import datetime

def fetch_fxstreet_feed(feed_url):
    """
    Parses an RSS feed and extracts relevant text data for sentiment analysis.
    """
    print(f"Fetching updates from: {feed_url}...")
    
    # Parse the RSS feed XML structure
    feed = feedparser.parse(feed_url)
    
    # Check for parsing or connection errors safely
    if feed.bozo:
        print("Warning: There was an issue parsing the feed. Data might be incomplete.")
        
    articles = []
    
    # Iterate through each news item in the feed
    for entry in feed.entries:
        # Extract core text fields required for your NLP pipeline
        title = entry.get('title', '')
        summary = entry.get('summary', '') or entry.get('description', '')
        link = entry.get('link', '')
        
        # Parse published date into a standard datetime object for time-series alignment
        published_parsed = entry.get('published_parsed')
        if published_parsed:
            timestamp = datetime(*published_parsed[:6])
        else:
            timestamp = datetime.now() # Fallback timestamp
            
        articles.append({
            'timestamp': timestamp,
            'title': title,
            'summary': summary,
            'link': link
        })
        
    # Convert list of dictionaries into a structured Pandas DataFrame
    df = pd.DataFrame(articles)
    
    # Ensure chronological order (oldest to newest) to match time-series forecasting direction
    if not df.empty:
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
    return df

if __name__ == "__main__":
    # Target URLs: FXStreet separates their technical analysis and raw macroeconomic news
    # Note: Replace these with the specific FXStreet RSS paths for Gold/Forex as needed
    FXSTREET_GOLD_NEWS_URL = "https://www.fxstreet.com/rss/news/commodities/gold" 
    FXSTREET_FOREX_NEWS_URL = "https://www.fxstreet.com/rss/news"
    
    try:
        # 1. Fetch the raw feed data
        news_df = fetch_fxstreet_feed(FXSTREET_GOLD_NEWS_URL)
        
        # 2. Preview the structured data
        print(f"\nSuccessfully retrieved {len(news_df)} articles.")
        print("\n--- Latest Article Preview ---")
        print(news_df)
        if not news_df.empty:
            print(f"Timestamp: {news_df['timestamp'].iloc[-1]}")
            print(f"Title:     {news_df['title'].iloc[-1]}")
            print(f"Summary:   {news_df['summary'].iloc[-1][:150]}...") # Truncated text preview
            
        # 3. Ready for your pipeline: 
        # You can now pass news_df['title'] or news_df['summary'] directly 
        # into your NLP text-embedding or sentiment scoring engine.
        
    except Exception as e:
        print(f"An error occurred executing the pipeline: {e}")