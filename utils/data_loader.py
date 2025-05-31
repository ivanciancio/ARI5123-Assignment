"""
Data Loader Module - EODHD Version

This module handles downloading and preprocessing financial data using EODHD API.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import logging
import time
import random
from typing import Dict, List, Optional
import streamlit as st

# Import the EODHD client
from utils.eodhdapi import EODHDClient, get_client

DATE_FORMAT = "%d/%m/%Y"

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, cache_dir="data"):
        """
        Initialise the DataLoader with EODHD client.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Initialise EODHD client
        try:
            self.eodhd_client = get_client()
            logger.info("Successfully initialised EODHD client")
        except Exception as e:
            logger.error(f"Failed to initialise EODHD client: {str(e)}")
            self.eodhd_client = None
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def format_ticker_for_eodhd(self, ticker: str) -> str:
        """
        Format ticker for EODHD API (add .US suffix for US stocks).
        
        Args:
            ticker: Original ticker symbol
            
        Returns:
            Formatted ticker for EODHD
        """
        # Special cases for indices
        if ticker == '^GSPC':
            return 'SPY.US'  # Use SPY ETF as proxy for S&P 500
        elif ticker in ['^FTSE', 'FTSE.INDX', 'UKX.INDX']:
            return 'ISF.LSE'  # Use iShares FTSE 100 ETF
        # Add .US suffix if not present
        elif '.' not in ticker:
            return f"{ticker}.US"
        return ticker
    
    def download_single_stock_eodhd(self, ticker: str, start: str, end: str, 
                                  max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Download data for a single stock using EODHD API.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame with stock data, or None if download fails
        """
        if self.eodhd_client is None:
            logger.error("EODHD client not initialised")
            return None
        
        # Format ticker for EODHD
        eodhd_ticker = self.format_ticker_for_eodhd(ticker)
        
        for attempt in range(max_retries):
            try:
                # Add small delay to respect rate limits
                if attempt > 0:
                    delay = 2 * (attempt + 1)  # 2, 4, 6 seconds
                    logger.info(f"Retrying {ticker} after {delay} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    # Small delay even for first attempt
                    time.sleep(0.5)
                
                # Download data using EODHD
                logger.info(f"Downloading {ticker} (as {eodhd_ticker}) from EODHD...")
                stock_data = self.eodhd_client.download(
                    symbol=eodhd_ticker,
                    start=start,
                    end=end,
                    interval='d'  # Daily data
                )
                
                # Check if download was successful
                if stock_data is None or stock_data.empty or len(stock_data) < 100:
                    logger.warning(f"Insufficient data for {ticker} (got {len(stock_data) if stock_data is not None else 0} rows)")
                    return None
                
                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in stock_data.columns for col in required_cols):
                    logger.error(f"Missing required columns for {ticker}")
                    return None
                
                # Keep only OHLCV data
                stock_data = stock_data[required_cols]
                
                # Clean data - remove any NaN values
                initial_len = len(stock_data)
                stock_data = stock_data.dropna()
                final_len = len(stock_data)
                
                if final_len < initial_len * 0.95:  # Lost more than 5% of data
                    logger.warning(f"Removed {initial_len - final_len} NaN rows from {ticker}")
                
                if len(stock_data) < 100:
                    logger.warning(f"Insufficient clean data for {ticker} after cleaning")
                    return None
                
                logger.info(f"Successfully downloaded {len(stock_data)} rows for {ticker}")
                return stock_data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                
                # For EODHD errors, wait a bit before retry
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 3))
        
        logger.error(f"Failed to download data for {ticker} after {max_retries} attempts")
        return None
    
    def download_benchmark_data(self, force_download=False, batch_size=10, progress_callback=None):
        """
        Download historical data for Dow 30 components using EODHD API.
        
        Args:
            force_download: Force fresh download even if cached data exists
            batch_size: Number of stocks to download in each batch (can be higher with EODHD)
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary of DataFrames with stock data
        """
        if self.eodhd_client is None:
            logger.error("EODHD client not initialised. Please check your API key.")
            if progress_callback:
                progress_callback(0, 1, "Error", "EODHD client not initialised")
            return {}
        
        # Use the complete benchmark paper's date range
        start_date = "01/01/1997"
        end_date = "31/12/2017"
        
        # Convert dates to EODHD format
        start = datetime.strptime(start_date, DATE_FORMAT).strftime("%Y-%m-%d")
        end = datetime.strptime(end_date, DATE_FORMAT).strftime("%Y-%m-%d")
        
        # Dow 30 components during the benchmark period
        dow30_tickers = [
            "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
            "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
            "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
        ]
        
        cache_file = os.path.join(self.cache_dir, f"benchmark_dow30_eodhd_{start}_{end}.pkl")
        
        # Return cached data if available
        if os.path.exists(cache_file) and not force_download:
            logger.info(f"Loading cached benchmark data from {cache_file}")
            try:
                cached_data = pd.read_pickle(cache_file)
                logger.info(f"Loaded {len(cached_data)} stocks from cache")
                
                # If progress callback provided, show instant completion
                if progress_callback:
                    for i, ticker in enumerate(dow30_tickers):
                        if ticker in cached_data:
                            progress_callback(i + 1, len(dow30_tickers), ticker, "✅ Cached")
                
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}. Downloading fresh data...")
        
        # Download fresh data
        logger.info(f"Downloading benchmark Dow 30 data from {start} to {end} using EODHD")
        logger.info(f"Using batch size of {batch_size} stocks")
        
        data = {}
        successful_downloads = 0
        failed_downloads = 0
        
        # Process stocks
        for i, ticker in enumerate(dow30_tickers):
            # Update progress - starting download
            if progress_callback:
                progress_callback(i, len(dow30_tickers), ticker, "downloading...")
            
            # Download with EODHD
            stock_data = self.download_single_stock_eodhd(ticker, start, end, max_retries=3)
            
            if stock_data is not None:
                data[ticker] = stock_data
                successful_downloads += 1
                logger.info(f"✅ {ticker}: {len(stock_data)} rows downloaded")
                
                if progress_callback:
                    progress_callback(i + 1, len(dow30_tickers), ticker, "✅ Success")
            else:
                failed_downloads += 1
                logger.error(f"❌ {ticker}: Download failed")
                if progress_callback:
                    progress_callback(i + 1, len(dow30_tickers), ticker, "❌ Failed")
            
            # EODHD has more generous rate limits, so we can use shorter delays
            if i < len(dow30_tickers) - 1:
                delay = random.uniform(0.5, 1.5)  # Much shorter delays
                time.sleep(delay)
            
            # Batch delay - also shorter for EODHD
            if (i + 1) % batch_size == 0 and i < len(dow30_tickers) - 1:
                batch_delay = random.uniform(2, 5)  # Shorter batch delays
                batch_num = (i + 1) // batch_size
                logger.info(f"Batch {batch_num} complete. Waiting {batch_delay:.1f} seconds before next batch...")
                
                if progress_callback:
                    progress_callback(
                        i + 1, 
                        len(dow30_tickers), 
                        f"Batch {batch_num}", 
                        f"Waiting {batch_delay:.1f}s...",
                        {"delay": batch_delay}
                    )
                    time.sleep(batch_delay)
        
        logger.info(f"Download complete: {successful_downloads}/{len(dow30_tickers)} stocks successful")
        
        # Cache the downloaded data even if incomplete
        if data:
            try:
                pd.to_pickle(data, cache_file)
                logger.info(f"Cached {len(data)} stocks to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        else:
            logger.error("No data was successfully downloaded!")
        
        return data
    
    def get_benchmark_splits(self, data, ticker, test_period="2007-2012"):
        """
        Get the exact benchmark splits used in Sezer & Ozbayoglu (2018).
        
        Args:
            data: Dictionary of DataFrames with stock data
            ticker: Stock ticker
            test_period: Either "2007-2012" or "2012-2017"
            
        Returns:
            Dictionary with train and test data splits
        """
        if ticker not in data:
            raise ValueError(f"Ticker {ticker} not found in data")
        
        df = data[ticker].copy()
        
        if test_period == "2007-2012":
            # First benchmark period
            train_start = "1997-01-01"
            train_end = "2006-12-31"
            test_start = "2007-01-01" 
            test_end = "2012-12-31"
        elif test_period == "2012-2017":
            # Second benchmark period  
            train_start = "1997-01-01"
            train_end = "2011-12-31"
            test_start = "2012-01-01"
            test_end = "2017-12-31"
        else:
            raise ValueError("test_period must be '2007-2012' or '2012-2017'")
        
        # Split the data
        train_data = df[train_start:train_end]
        test_data = df[test_start:test_end]
        
        logger.info(f"Benchmark split for {ticker} ({test_period}):")
        logger.info(f"  Training: {len(train_data)} samples ({train_start} to {train_end})")
        logger.info(f"  Testing: {len(test_data)} samples ({test_start} to {test_end})")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'train_dates': (train_start, train_end),
            'test_dates': (test_start, test_end),
            'ticker': ticker,
            'test_period': test_period
        }
    
    def get_test_dates(self, data, ticker, window_size=30, train_ratio=0.7, val_ratio=0.15):
        """
        Legacy method for backward compatibility.
        """
        if ticker not in data:
            raise ValueError(f"Ticker {ticker} not found in data")
        
        df = data[ticker].copy()
        total_samples = len(df) - window_size
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        test_start_idx = train_size + val_size + window_size
        test_dates = df.index[test_start_idx:]
        
        return test_dates