"""
Data Loader Module

This module handles downloading and preprocessing financial data for the trading model.
The implementation follows the approach in Sezer & Ozbayoglu (2018) with proper benchmark dates.
Includes robust rate limiting and retry logic for Yahoo Finance API.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import logging
import time
import random
from typing import Dict, List, Optional

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
    
    def download_single_stock_with_retry(self, ticker: str, start: str, end: str, 
                                       max_retries: int = 3, base_delay: float = 1.0) -> Optional[pd.DataFrame]:
        """
        Download data for a single stock with retry logic and rate limiting.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between requests in seconds
            
        Returns:
            DataFrame with stock data, or None if download fails
        """
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid hitting rate limits
                delay = base_delay * (2 ** attempt) + random.uniform(2.0, 5.0)
                if attempt > 0:
                    logger.info(f"Retrying {ticker} after {delay:.1f} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    # Add delay even for first attempt
                    time.sleep(random.uniform(3.0, 5.0))  # New: delay for first attempt
                
                # Download data for single ticker
                logger.info(f"Downloading {ticker}...")
                stock_data = yf.download(
                    ticker, 
                    start=start, 
                    end=end, 
                    progress=False,  # Disable progress bar to reduce API calls
                    auto_adjust=True,  # Handle splits and dividends
                    prepost=False,  # Only trading hours
                    threads=False   # Single threaded to avoid rate limits
                )
                
                # Check if download was successful
                if stock_data.empty or len(stock_data) < 100:
                    logger.warning(f"Insufficient data for {ticker} (got {len(stock_data)} rows)")
                    return None
                
                # Keep only OHLCV data and clean column names
                if isinstance(stock_data.columns, pd.MultiIndex):
                    # Handle multi-level columns from yfinance
                    stock_data = stock_data.droplevel(1, axis=1)
                
                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in stock_data.columns for col in required_cols):
                    logger.error(f"Missing required columns for {ticker}")
                    return None
                
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
                
                # Check if it's a rate limit error
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff for rate limits
                        delay = base_delay * (3 ** attempt) + random.uniform(2, 5)
                        logger.info(f"Rate limited. Waiting {delay:.1f} seconds before retry...")
                        time.sleep(delay)
                        continue
                
                # For other errors, wait a bit but don't use exponential backoff
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 3))
        
        logger.error(f"Failed to download data for {ticker} after {max_retries} attempts")
        return None
    
    def download_benchmark_data(self, force_download=False, batch_size=5, progress_callback=None):
        """
        Download historical data for Dow 30 components using the exact benchmark date range
        from Sezer & Ozbayoglu (2018): 1997-2017 for complete coverage.
        
        Args:
            force_download: Force fresh download even if cached data exists
            batch_size: Number of stocks to download in each batch (smaller = slower but more reliable)
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary of DataFrames with stock data
        """
        # Use the complete benchmark paper's date range for both training periods
        start_date = "01/01/1997"  # Extended range to match paper
        end_date = "31/12/2017"    # Benchmark end date
        
        # Convert dates to yfinance format
        start = datetime.strptime(start_date, DATE_FORMAT).strftime("%Y-%m-%d")
        end = datetime.strptime(end_date, DATE_FORMAT).strftime("%Y-%m-%d")
        
        # Dow 30 components during the benchmark period (exact same as paper)
        dow30_tickers = [
            "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
            "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
            "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
        ]
        
        cache_file = os.path.join(self.cache_dir, f"benchmark_dow30_{start}_{end}.pkl")
        
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
        logger.info(f"Downloading benchmark Dow 30 data from {start} to {end}")
        logger.info(f"Using batch size of {batch_size} stocks to avoid rate limits")
        
        data = {}
        successful_downloads = 0
        failed_downloads = 0
        
        # Process stocks
        for i, ticker in enumerate(dow30_tickers):
            # Update progress - starting download
            if progress_callback:
                progress_callback(i, len(dow30_tickers), ticker, "downloading...")
            
            # Download with enhanced retry logic
            stock_data = self.download_single_stock_with_retry_enhanced(
                ticker, start, end, max_retries=3, base_delay=2.0, 
                progress_callback=progress_callback, 
                current_index=i, 
                total_tickers=len(dow30_tickers)
            )
            
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
            
            # Add delay between individual downloads within batch
            if i < len(dow30_tickers) - 1:
                delay = random.uniform(1.5, 3.0)
                time.sleep(delay)
            
            # Add longer delay between batches
            if (i + 1) % batch_size == 0 and i < len(dow30_tickers) - 1:
                batch_delay = random.uniform(5, 10)
                batch_num = (i + 1) // batch_size
                logger.info(f"Batch {batch_num} complete. Waiting {batch_delay:.1f} seconds before next batch...")
                
                if progress_callback:
                    for wait_second in range(int(batch_delay)):
                        remaining = int(batch_delay - wait_second)
                        progress_callback(
                            i + 1, 
                            len(dow30_tickers), 
                            f"Batch {batch_num}", 
                            f"Waiting {remaining}s to avoid rate limit...",
                            {"delay": remaining}
                        )
                        time.sleep(1)
        
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
    
    def download_single_stock_with_retry_enhanced(self, ticker: str, start: str, end: str, 
                                            max_retries: int = 3, base_delay: float = 1.0,
                                            progress_callback=None, current_index=0, 
                                            total_tickers=1) -> Optional[pd.DataFrame]:
        """
        Enhanced version of download_single_stock_with_retry with progress callback support.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between requests in seconds
            progress_callback: Optional callback for progress updates
            current_index: Current ticker index for progress tracking
            total_tickers: Total number of tickers for progress tracking
            
        Returns:
            DataFrame with stock data, or None if download fails
        """
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid hitting rate limits
                delay = base_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                if attempt > 0:
                    logger.info(f"Retrying {ticker} after {delay:.1f} seconds (attempt {attempt + 1}/{max_retries})")
                    
                    # Update progress with retry status
                    if progress_callback:
                        progress_callback(
                            current_index, 
                            total_tickers, 
                            ticker, 
                            f"Retrying (attempt {attempt + 1}/{max_retries})...",
                            {"delay": delay}
                        )
                        
                    time.sleep(delay)
                
                # Download data for single ticker
                logger.info(f"Downloading {ticker}...")
                stock_data = yf.download(
                    ticker, 
                    start=start, 
                    end=end, 
                    progress=False,  # Disable progress bar to reduce API calls
                    auto_adjust=True,  # Handle splits and dividends
                    prepost=False,  # Only trading hours
                    threads=False   # Single threaded to avoid rate limits
                )
                
                # Check if download was successful
                if stock_data.empty or len(stock_data) < 100:
                    logger.warning(f"Insufficient data for {ticker} (got {len(stock_data)} rows)")
                    return None
                
                # Keep only OHLCV data and clean column names
                if isinstance(stock_data.columns, pd.MultiIndex):
                    # Handle multi-level columns from yfinance
                    stock_data = stock_data.droplevel(1, axis=1)
                
                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in stock_data.columns for col in required_cols):
                    logger.error(f"Missing required columns for {ticker}")
                    return None
                
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
                
                # Check if it's a rate limit error
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff for rate limits
                        delay = base_delay * (3 ** attempt) + random.uniform(2, 5)
                        logger.info(f"Rate limited. Waiting {delay:.1f} seconds before retry...")
                        time.sleep(delay)
                        continue
                
                # For other errors, wait a bit but don't use exponential backoff
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 3))
        
        logger.error(f"Failed to download data for {ticker} after {max_retries} attempts")
        return None
    
    def download_stocks_sequentially(self, tickers: List[str], start: str, end: str, 
                                   delay_range: tuple = (2, 5)) -> Dict[str, pd.DataFrame]:
        """
        Download multiple stocks sequentially with delays to avoid rate limiting.
        
        Args:
            tickers: List of stock tickers
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            delay_range: Tuple of (min_delay, max_delay) in seconds between downloads
            
        Returns:
            Dictionary of DataFrames with stock data
        """
        data = {}
        
        for i, ticker in enumerate(tickers):
            logger.info(f"Downloading {ticker} ({i+1}/{len(tickers)})")
            
            stock_data = self.download_single_stock_with_retry(ticker, start, end)
            
            if stock_data is not None:
                data[ticker] = stock_data
            
            # Add delay between downloads (except for the last one)
            if i < len(tickers) - 1:
                delay = random.uniform(delay_range[0], delay_range[1])
                logger.info(f"Waiting {delay:.1f} seconds before next download...")
                time.sleep(delay)
        
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