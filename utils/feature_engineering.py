"""
Feature Engineering Module

This module implements the image creation and labeling process based on Sezer & Ozbayoglu (2018).
"""

import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class to create bar chart images and labels for the CNN-BI trading model."""
    
    def __init__(self):
        """Initialise the FeatureEngineer."""
        pass
    
    def create_bar_chart_image(self, price_data, window_size=30):
        """
        Create a 30x30 binary image from a window of price data.
        This follows the approach in Sezer & Ozbayoglu (2018) paper.
        
        Args:
            price_data: Array of price values for the window
            window_size: Size of the window (default: 30)
            
        Returns:
            30x30 numpy array containing the binary image
        """
        # Create empty image
        img = np.zeros((30, 30))
        
        # Normalise prices to fit in image
        min_val = np.min(price_data)
        max_val = np.max(price_data)
        price_range = max_val - min_val
        
        if price_range > 0:  # Avoid division by zero
            for i in range(min(window_size, len(price_data))):
                # Calculate normalised bar height (0-29)
                normalized_price = int(29 * (price_data[i] - min_val) / price_range)
                
                # Draw bar from bottom to height (filling the column)
                # In the binary image, 1 represents the bar, 0 is background
                img[29-normalized_price:30, i] = 1
        
        return img
    
    def create_image_dataset(self, data, ticker, window_size=30):
        """
        Create a dataset of bar chart images from stock price data.
        
        Args:
            data: Dictionary of DataFrames with stock data
            ticker: Stock ticker to process
            window_size: Size of the window for each image
            
        Returns:
            Tuple of (images, dates) arrays
        """
        if ticker not in data:
            raise ValueError(f"Ticker {ticker} not found in data")
        
        # Get price data for the ticker
        df = data[ticker].copy()
        close_prices = df['Close'].values
        
        # Create images
        images = []
        dates = []
        
        for i in range(len(close_prices) - window_size + 1):
            # Get window of prices
            window = close_prices[i:i+window_size]
            
            # Create image
            img = self.create_bar_chart_image(window, window_size)
            
            # Add to dataset
            images.append(img)
            dates.append(df.index[i+window_size-1])
        
        return np.array(images), np.array(dates)
    
    def generate_labels(self, data, ticker, window_size=30):
        """
        Generate labels for bar chart images based on price slopes.
        Follows the approach in Sezer & Ozbayoglu (2018) paper.
        
        Args:
            data: Dictionary of DataFrames with stock data
            ticker: Stock ticker to process
            window_size: Size of the window for each image
            
        Returns:
            Array of labels (0=Hold, 1=Buy, 2=Sell)
        """
        if ticker not in data:
            raise ValueError(f"Ticker {ticker} not found in data")
        
        # Get price data for the ticker
        df = data[ticker].copy()
        close_prices = df['Close'].values
        
        # Calculate slopes for each window
        slopes = []
        
        for i in range(len(close_prices) - window_size - 15):  # Need 15 days for far-future
            day30_price = close_prices[i+window_size-1]  # Last day in the window
            day45_price = close_prices[i+window_size+14]  # 15 days in the future
            
            # Calculate slope as percentage change
            slope = (day45_price - day30_price) / (day30_price * 15)
            slopes.append(slope)
        
        # Create distribution of slopes
        slope_array = np.array(slopes)
        
        # Calculate thresholds from distribution (as per paper)
        threshold_high = np.percentile(slope_array, 60)  # Top 40% for Buy
        threshold_low = np.percentile(slope_array, 40)   # Bottom 40% for Sell
        
        logger.info(f"Generated slope thresholds for {ticker}: high={threshold_high:.4f}, low={threshold_low:.4f}")
        
        # Generate labels
        labels = []
        
        for i in range(len(close_prices) - window_size - 15):
            day30_price = close_prices[i+window_size-1]  # Last day in the window
            day45_price = close_prices[i+window_size+14]  # 15 days in the future
            
            # Calculate current slope
            slope = (day45_price - day30_price) / (day30_price * 15)
            
            # Determine label based on thresholds
            if slope > threshold_high:
                label = 1  # Buy
            elif slope < threshold_low:
                label = 2  # Sell
            else:
                label = 0  # Hold
            
            labels.append(label)
        
        return np.array(labels)
    
    def prepare_image_data_for_model(self, data, ticker, window_size=30, 
                                    train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        Prepare image data and labels for the CNN model, split into train/val/test sets.
        
        Args:
            data: Dictionary of DataFrames with stock data
            ticker: Stock ticker to process
            window_size: Size of the window for each image
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with training, validation, and test data
        """
        # Create images and generate labels
        images, dates = self.create_image_dataset(data, ticker, window_size)
        labels = self.generate_labels(data, ticker, window_size)
        
        # Ensure images and labels have the same length
        # Labels are shorter because we need future data to calculate them
        num_labels = len(labels)
        images = images[:num_labels]
        dates = dates[:num_labels]
        
        # Split data into train, validation, and test sets
        # We use time-based splitting instead of random to preserve time series structure
        train_size = int(num_labels * train_ratio)
        val_size = int(num_labels * val_ratio)
        
        # Training set
        X_train = images[:train_size]
        y_train = labels[:train_size]
        train_dates = dates[:train_size]
        
        # Validation set
        X_val = images[train_size:train_size+val_size]
        y_val = labels[train_size:train_size+val_size]
        val_dates = dates[train_size:train_size+val_size]
        
        # Test set
        X_test = images[train_size+val_size:]
        y_test = labels[train_size+val_size:]
        test_dates = dates[train_size+val_size:]
        
        logger.info(f"Prepared data for {ticker}: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test")
        
        # Return prepared data
        return {
            'X_train': X_train,
            'y_train': y_train,
            'train_dates': train_dates,
            'X_val': X_val,
            'y_val': y_val,
            'val_dates': val_dates,
            'X_test': X_test,
            'y_test': y_test,
            'test_dates': test_dates,
            'ticker': ticker,
            'window_size': window_size
        }
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe.
        This can be used for alternative models or visualisation.
        
        Args:
            df: Pandas DataFrame with OHLCV data
                
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # 1. TREND INDICATORS
        # Moving Averages
        df_features['SMA_5'] = ta.trend.sma_indicator(df_features['Close'], window=5)
        df_features['SMA_20'] = ta.trend.sma_indicator(df_features['Close'], window=20)
        df_features['SMA_50'] = ta.trend.sma_indicator(df_features['Close'], window=50)
        
        # Exponential Moving Averages
        df_features['EMA_10'] = ta.trend.ema_indicator(df_features['Close'], window=10)
        
        # MACD
        macd = ta.trend.MACD(df_features['Close'], window_fast=12, window_slow=26, window_sign=9)
        df_features['MACD'] = macd.macd()
        df_features['MACD_Signal'] = macd.macd_signal()
        df_features['MACD_Diff'] = macd.macd_diff()
        
        # 2. MOMENTUM INDICATORS
        # RSI
        df_features['RSI_14'] = ta.momentum.RSIIndicator(df_features['Close'], window=14).rsi()
        
        # 3. VOLATILITY INDICATORS
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df_features['Close'])
        df_features['BB_High'] = bollinger.bollinger_hband()
        df_features['BB_Low'] = bollinger.bollinger_lband()
        df_features['BB_Mid'] = bollinger.bollinger_mavg()
        
        # Average True Range
        df_features['ATR_14'] = ta.volatility.AverageTrueRange(df_features['High'], df_features['Low'], df_features['Close'], window=14).average_true_range()
        
        # Drop NaN values resulting from indicators that use lookback windows
        df_features = df_features.dropna()
        
        return df_features
    
    def prepare_features_for_model(self, data, ticker, add_indicators=True, add_sequences=False):
        """
        Prepare features for the specified ticker.
        This is used for traditional ML approaches (not CNN-BI).
        
        Args:
            data: Dictionary of DataFrames with stock data
            ticker: Stock ticker to process
            add_indicators: Whether to add technical indicators
            add_sequences: Whether to add sequence features
            
        Returns:
            DataFrame with all features
        """
        if ticker not in data:
            raise ValueError(f"Ticker {ticker} not found in data")
        
        df = data[ticker].copy()
        
        # Add technical indicators if requested
        if add_indicators:
            df = self.add_technical_indicators(df)
        
        return df