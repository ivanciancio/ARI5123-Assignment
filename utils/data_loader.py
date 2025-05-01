"""
Data Loader Module

This module handles downloading and preprocessing financial data for the trading model.
The implementation follows the approach in Sezer & Ozbayoglu (2018) with proper benchmark dates.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import logging

DATE_FORMAT = "%d/%m/%Y"

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, cache_dir="data"):
        """
        Initialise the DataLoader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def download_dow30_data(self, start_date=None, end_date=None, force_download=False):
        """
        Download historical data for Dow 30 components using the benchmark date range
        from Sezer & Ozbayoglu (2018) by default.
        
        Args:
            start_date: Start date in DD/MM/YYYY format (defaults to 01/01/2007)
            end_date: End date in DD/MM/YYYY format (defaults to 31/12/2017)
            force_download: Force fresh download even if cached data exists
            
        Returns:
            Dictionary of DataFrames with stock data
        """
        # Use the benchmark paper's date range if not specified
        if start_date is None:
            start_date = "01/01/2007"  # Benchmark start date
        if end_date is None:
            end_date = "31/12/2017"  # Benchmark end date
        
        # Convert dates to yfinance format
        start = datetime.strptime(start_date, DATE_FORMAT).strftime("%Y-%m-%d")
        end = datetime.strptime(end_date, DATE_FORMAT).strftime("%Y-%m-%d")
        
        # Dow 30 components during the 2007-2017 period
        # This reflects the major components during most of the benchmark period
        # Note: There were some changes within this period
        dow30_tickers = [
            "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
            "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
            "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
        ]
        
        cache_file = os.path.join(self.cache_dir, f"dow30_{start}_{end}.pkl")
        
        # Return cached data if available
        if os.path.exists(cache_file) and not force_download:
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)
        
        # Download fresh data
        logger.info(f"Downloading Dow 30 data from {start} to {end}")
        data = {}
        
        for ticker in dow30_tickers:
            try:
                # Download data
                stock_data = yf.download(ticker, start=start, end=end)
                
                # Skip if no data or very limited data
                if len(stock_data) < 30:
                    logger.warning(f"Insufficient data for {ticker}, skipping")
                    continue
                
                # Keep only OHLCV data
                stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Add ticker as a column and rename columns to British English naming
                stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                data[ticker] = stock_data
                logger.info(f"Downloaded {len(stock_data)} rows for {ticker}")
                
            except Exception as e:
                logger.error(f"Error downloading data for {ticker}: {e}")
        
        # Cache the downloaded data
        if data:
            pd.to_pickle(data, cache_file)
            logger.info(f"Cached data to {cache_file}")
        
        return data
    
    def get_test_dates(self, data, ticker, window_size=30, train_ratio=0.7, val_ratio=0.15):
        """
        Get the dates corresponding to the test set.
        
        Args:
            data: Dictionary of DataFrames with stock data
            ticker: Stock ticker
            window_size: Window size for sequence generation
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
        
        Returns:
            Array of dates for the test set
        """
        if ticker not in data:
            raise ValueError(f"Ticker {ticker} not found in data")
        
        # Get the DataFrame for the ticker
        df = data[ticker].copy()
        
        # Calculate dataset sizes
        total_samples = len(df) - window_size
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        # Get test set dates
        test_start_idx = train_size + val_size + window_size
        test_dates = df.index[test_start_idx:]
        
        return test_dates