import yfinance as yf
import pandas as pd

def fetch_gold_candles(ticker_symbol="GC=F", interval="5m", count=1000):
    """
    Fetches the latest intraday candles for Gold (XAU/USD) using yfinance.
    
    Parameters:
    - ticker_symbol: 'GC=F' represents Gold continuous futures.
    - interval: '5m' for 5-minute candlesticks.
    - count: The number of recent trailing candles to return.
    """
    print(f"Requesting 5-minute interval data for {ticker_symbol}...")
    
    # 60 days is the maximum allowed window for 5m data on Yahoo Finance
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="60d", interval=interval)
    
    if df.empty:
        print("Error: No data retrieved. Verify network connectivity or ticker.")
        return pd.DataFrame()
    
    # Sort chronologically just in case, and slice the last N candles
    df = df.sort_index()
    latest_candles = df.tail(count)
    
    # Clean up columns for your multi-model feature fusion pipeline
    latest_candles = latest_candles[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return latest_candles

if __name__ == "__main__":
    # Fetch 1000 candles of 5m interval data
    gold_data = fetch_gold_candles(ticker_symbol="GC=F", interval="5m", count=1000)
    
    if not gold_data.empty:
        print(f"\nSuccessfully fetched {len(gold_data)} candles!")
        print("\n--- First 3 Rows (Oldest in the 1000 slice) ---")
        print(gold_data.head(3))
        
        print("\n--- Last 3 Rows (Most Recent Market Status) ---")
        print(gold_data.tail(3))
        
        # Optional: Save to CSV to checkpoint your dataset
        gold_data.to_csv("XAUUSD_5m_1000_candles.csv")
        print("\nDataset successfully cached to 'XAUUSD_5m_1000_candles.csv'")