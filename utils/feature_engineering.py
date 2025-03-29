"""
Feature Engineering Module

This module calculates technical indicators used as features for the trading model.
These indicators enhance the simple OHLCV data used in the original paper.
"""

import pandas as pd
import numpy as np
import ta


class FeatureEngineer:
    """Class to calculate technical indicators and generate features for trading models."""
    
    def __init__(self):
        """Initialise the FeatureEngineer."""
        pass
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df_features.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # 1. Trend Indicators
        # Moving Averages
        df_features['SMA_10'] = ta.trend.sma_indicator(df_features['Close'], window=10)
        df_features['SMA_20'] = ta.trend.sma_indicator(df_features['Close'], window=20)
        df_features['SMA_50'] = ta.trend.sma_indicator(df_features['Close'], window=50)
        
        # MACD
        macd = ta.trend.MACD(df_features['Close'])
        df_features['MACD'] = macd.macd()
        df_features['MACD_Signal'] = macd.macd_signal()
        df_features['MACD_Diff'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df_features['High'], df_features['Low'], df_features['Close'])
        df_features['ADX'] = adx.adx()
        df_features['ADX_Pos'] = adx.adx_pos()
        df_features['ADX_Neg'] = adx.adx_neg()
        
        # 2. Momentum Indicators
        # RSI (Relative Strength Index)
        df_features['RSI'] = ta.momentum.RSIIndicator(df_features['Close']).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df_features['High'], df_features['Low'], df_features['Close'])
        df_features['Stoch_K'] = stoch.stoch()
        df_features['Stoch_D'] = stoch.stoch_signal()
        
        # 3. Volatility Indicators
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df_features['Close'])
        df_features['BB_High'] = bollinger.bollinger_hband()
        df_features['BB_Low'] = bollinger.bollinger_lband()
        df_features['BB_Mid'] = bollinger.bollinger_mavg()
        df_features['BB_Width'] = df_features['BB_High'] - df_features['BB_Low']
        df_features['BB_Pct'] = (df_features['Close'] - df_features['BB_Low']) / (df_features['BB_High'] - df_features['BB_Low'])
        
        # ATR (Average True Range)
        df_features['ATR'] = ta.volatility.AverageTrueRange(df_features['High'], df_features['Low'], df_features['Close']).average_true_range()
        
        # 4. Volume Indicators
        # Volume moving average - FIXED: using ta.trend instead of ta.volatility
        df_features['Volume_SMA_10'] = ta.trend.sma_indicator(df_features['Volume'], window=10)
        
        # On-Balance Volume
        df_features['OBV'] = ta.volume.OnBalanceVolumeIndicator(df_features['Close'], df_features['Volume']).on_balance_volume()
        
        # 5. Price transformations and ratios
        # Price change percentage
        df_features['Price_Change_1d'] = df_features['Close'].pct_change(1)
        df_features['Price_Change_5d'] = df_features['Close'].pct_change(5)
        df_features['Price_Change_10d'] = df_features['Close'].pct_change(10)
        
        # Normalised price ratios
        df_features['Close_to_Open'] = df_features['Close'] / df_features['Open']
        df_features['High_to_Low'] = df_features['High'] / df_features['Low']
        
        # 6. Candlestick patterns - represented numerically
        df_features['Body_Size'] = abs(df_features['Close'] - df_features['Open']) / (df_features['High'] - df_features['Low'])
        df_features['Upper_Shadow'] = (df_features['High'] - np.maximum(df_features['Open'], df_features['Close'])) / (df_features['High'] - df_features['Low'])
        df_features['Lower_Shadow'] = (np.minimum(df_features['Open'], df_features['Close']) - df_features['Low']) / (df_features['High'] - df_features['Low'])
        
        # Drop NaN values
        df_features = df_features.dropna()
        
        return df_features
    
    def add_sequence_features(self, df, window_size=20):
        """
        Create features that capture sequential patterns over a window.
        
        Args:
            df: Pandas DataFrame with OHLCV and indicator data
            window_size: Size of the window for sequential features
            
        Returns:
            DataFrame with added sequence features
        """
        df_seq = df.copy()
        
        # Trend direction features
        df_seq['Uptrend'] = ((df_seq['Close'] > df_seq['SMA_20']) & 
                             (df_seq['SMA_20'] > df_seq['SMA_50'])).astype(int)
        
        # Momentum regime features
        df_seq['Momentum_Regime'] = ((df_seq['RSI'] > 50) & 
                                     (df_seq['MACD'] > 0)).astype(int)
        
        # Volatility regime features
        df_seq['High_Volatility'] = (df_seq['ATR'] > 
                                     df_seq['ATR'].rolling(window=window_size).mean()).astype(int)
        
        # Calculate streak features (consecutive up/down days)
        df_seq['Up_Day'] = (df_seq['Close'] > df_seq['Close'].shift(1)).astype(int)
        df_seq['Up_Streak'] = df_seq['Up_Day'].groupby((df_seq['Up_Day'] != df_seq['Up_Day'].shift(1)).cumsum()).cumcount() + 1
        df_seq['Down_Streak'] = df_seq['Up_Day'].replace({1: 0, 0: 1}).groupby((df_seq['Up_Day'] != df_seq['Up_Day'].shift(1)).cumsum()).cumcount() + 1
        
        # Calculate mean reversion potential
        df_seq['Dist_From_SMA50_Pct'] = (df_seq['Close'] - df_seq['SMA_50']) / df_seq['SMA_50']
        df_seq['Overbought'] = (df_seq['RSI'] > 70).astype(int)
        df_seq['Oversold'] = (df_seq['RSI'] < 30).astype(int)
        
        # Create a composite signal feature
        df_seq['Composite_Signal'] = (df_seq['Uptrend'] * 0.3 + 
                                     df_seq['Momentum_Regime'] * 0.3 + 
                                     df_seq['Overbought'] * -0.2 + 
                                     df_seq['Oversold'] * 0.2)
        
        return df_seq
    
    def prepare_features_for_model(self, data, ticker, add_indicators=True, add_sequences=True):
        """
        Prepare features for the specified ticker.
        
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
        
        # Add sequence features if requested
        if add_sequences:
            df = self.add_sequence_features(df)
        
        return df