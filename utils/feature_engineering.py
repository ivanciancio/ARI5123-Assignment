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
        Add enhanced technical indicators to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
                
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # 1. TREND INDICATORS
        # Moving Averages - additional periods
        df_features['SMA_5'] = ta.trend.sma_indicator(df_features['Close'], window=5)
        df_features['SMA_10'] = ta.trend.sma_indicator(df_features['Close'], window=10)
        df_features['SMA_20'] = ta.trend.sma_indicator(df_features['Close'], window=20)
        df_features['SMA_50'] = ta.trend.sma_indicator(df_features['Close'], window=50)
        df_features['SMA_100'] = ta.trend.sma_indicator(df_features['Close'], window=100)
        
        # Exponential Moving Averages
        df_features['EMA_5'] = ta.trend.ema_indicator(df_features['Close'], window=5)
        df_features['EMA_10'] = ta.trend.ema_indicator(df_features['Close'], window=10)
        df_features['EMA_20'] = ta.trend.ema_indicator(df_features['Close'], window=20)
        
        # MACD with different parameters
        macd = ta.trend.MACD(df_features['Close'], window_fast=12, window_slow=26, window_sign=9)
        df_features['MACD'] = macd.macd()
        df_features['MACD_Signal'] = macd.macd_signal()
        df_features['MACD_Diff'] = macd.macd_diff()
        
        # Parabolic SAR
        df_features['PSAR'] = ta.trend.PSARIndicator(df_features['High'], df_features['Low'], df_features['Close']).psar()
        
        # ADX with different parameters
        adx = ta.trend.ADXIndicator(df_features['High'], df_features['Low'], df_features['Close'], window=14)
        df_features['ADX'] = adx.adx()
        df_features['ADX_Pos'] = adx.adx_pos()
        df_features['ADX_Neg'] = adx.adx_neg()
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df_features['High'], df_features['Low'])
        df_features['Ichimoku_A'] = ichimoku.ichimoku_a()
        df_features['Ichimoku_B'] = ichimoku.ichimoku_b()
        
        # 2. MOMENTUM INDICATORS
        # RSI with multiple timeframes
        df_features['RSI_14'] = ta.momentum.RSIIndicator(df_features['Close'], window=14).rsi()
        df_features['RSI_7'] = ta.momentum.RSIIndicator(df_features['Close'], window=7).rsi()
        df_features['RSI_21'] = ta.momentum.RSIIndicator(df_features['Close'], window=21).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df_features['High'], df_features['Low'], df_features['Close'])
        df_features['Stoch_K'] = stoch.stoch()
        df_features['Stoch_D'] = stoch.stoch_signal()
        
        # Williams %R
        df_features['Williams_R'] = ta.momentum.WilliamsRIndicator(df_features['High'], df_features['Low'], df_features['Close']).williams_r()
        
        # Rate of Change
        df_features['ROC_10'] = ta.momentum.ROCIndicator(df_features['Close'], window=10).roc()
        
        # 3. VOLATILITY INDICATORS
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df_features['Close'])
        df_features['BB_High'] = bollinger.bollinger_hband()
        df_features['BB_Low'] = bollinger.bollinger_lband()
        df_features['BB_Mid'] = bollinger.bollinger_mavg()
        df_features['BB_Width'] = (df_features['BB_High'] - df_features['BB_Low']) / df_features['BB_Mid']
        df_features['BB_Pct'] = (df_features['Close'] - df_features['BB_Low']) / (df_features['BB_High'] - df_features['BB_Low'])
        
        # Average True Range with multiple timeframes
        df_features['ATR_14'] = ta.volatility.AverageTrueRange(df_features['High'], df_features['Low'], df_features['Close'], window=14).average_true_range()
        df_features['ATR_7'] = ta.volatility.AverageTrueRange(df_features['High'], df_features['Low'], df_features['Close'], window=7).average_true_range()
        
        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(df_features['High'], df_features['Low'], df_features['Close'])
        df_features['KC_High'] = keltner.keltner_channel_hband()
        df_features['KC_Low'] = keltner.keltner_channel_lband()
        
        # 4. VOLUME INDICATORS
        # Volume moving average
        df_features['Volume_SMA_10'] = ta.trend.sma_indicator(df_features['Volume'], window=10)
        df_features['Volume_SMA_20'] = ta.trend.sma_indicator(df_features['Volume'], window=20)
        
        # On-Balance Volume
        df_features['OBV'] = ta.volume.OnBalanceVolumeIndicator(df_features['Close'], df_features['Volume']).on_balance_volume()
        
        # Volume Rate of Change
        df_features['Volume_ROC'] = ta.momentum.ROCIndicator(df_features['Volume'], window=10).roc()
        
        # Chaikin Money Flow
        df_features['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df_features['High'], df_features['Low'], df_features['Close'], df_features['Volume']).chaikin_money_flow()
        
        # 5. PRICE TRANSFORMATIONS AND RATIOS
        # Price change percentage across multiple timeframes
        df_features['Price_Change_1d'] = df_features['Close'].pct_change(1)
        df_features['Price_Change_3d'] = df_features['Close'].pct_change(3)
        df_features['Price_Change_5d'] = df_features['Close'].pct_change(5)
        df_features['Price_Change_10d'] = df_features['Close'].pct_change(10)
        
        # Log returns
        df_features['Log_Return_1d'] = np.log(df_features['Close'] / df_features['Close'].shift(1))
        
        # Volatility (standard deviation of returns)
        df_features['Volatility_5d'] = df_features['Log_Return_1d'].rolling(window=5).std()
        df_features['Volatility_10d'] = df_features['Log_Return_1d'].rolling(window=10).std()
        df_features['Volatility_20d'] = df_features['Log_Return_1d'].rolling(window=20).std()
        
        # Normalized price ratios
        df_features['Close_to_Open'] = df_features['Close'] / df_features['Open']
        df_features['High_to_Low'] = df_features['High'] / df_features['Low']
        df_features['Close_to_SMA20'] = df_features['Close'] / df_features['SMA_20']
        df_features['Close_to_SMA50'] = df_features['Close'] / df_features['SMA_50']
        
        # 6. CANDLESTICK PATTERNS
        df_features['Body_Size'] = abs(df_features['Close'] - df_features['Open']) / (df_features['High'] - df_features['Low'])
        df_features['Upper_Shadow'] = (df_features['High'] - np.maximum(df_features['Open'], df_features['Close'])) / (df_features['High'] - df_features['Low'])
        df_features['Lower_Shadow'] = (np.minimum(df_features['Open'], df_features['Close']) - df_features['Low']) / (df_features['High'] - df_features['Low'])
        
        # 7. MARKET REGIME FEATURES
        # Trend strength indicator (ADX > 25 indicates strong trend)
        df_features['Strong_Trend'] = (df_features['ADX'] > 25).astype(int)
        
        # Market volatility regime
        df_features['High_Volatility'] = (df_features['ATR_14'] > df_features['ATR_14'].rolling(window=20).mean()).astype(int)
        
        # Price momentum regime (RSI > 50 and MACD > 0 indicates bullish momentum)
        df_features['Bullish_Momentum'] = ((df_features['RSI_14'] > 50) & (df_features['MACD'] > 0)).astype(int)
        
        # Price mean-reversion indicator (oversold or overbought)
        df_features['Oversold'] = (df_features['RSI_14'] < 30).astype(int)
        df_features['Overbought'] = (df_features['RSI_14'] > 70).astype(int)
        
        # Feature normalization - Z-score for appropriate features
        price_features = ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'EMA_5', 'EMA_10', 'EMA_20']
        for feature in price_features:
            if feature in df_features.columns:
                # Center around close price
                df_features[f'{feature}_Ratio'] = df_features['Close'] / df_features[feature]
        
        # Drop NaN values resulting from indicators that use lookback windows
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
        
        # Check if required columns exist before using them
        if 'SMA_20' in df_seq.columns and 'SMA_50' in df_seq.columns:
            # Trend direction features
            df_seq['Uptrend'] = ((df_seq['Close'] > df_seq['SMA_20']) & 
                                (df_seq['SMA_20'] > df_seq['SMA_50'])).astype(int)
        
        if 'RSI_14' in df_seq.columns and 'MACD' in df_seq.columns:
            # Momentum regime features
            df_seq['Momentum_Regime'] = ((df_seq['RSI_14'] > 50) & 
                                        (df_seq['MACD'] > 0)).astype(int)
        
        if 'ATR_14' in df_seq.columns:
            # Volatility regime features
            df_seq['High_Volatility'] = (df_seq['ATR_14'] > 
                                        df_seq['ATR_14'].rolling(window=window_size).mean()).astype(int)
        
        # Calculate streak features (consecutive up/down days)
        df_seq['Up_Day'] = (df_seq['Close'] > df_seq['Close'].shift(1)).astype(int)
        df_seq['Up_Streak'] = df_seq['Up_Day'].groupby((df_seq['Up_Day'] != df_seq['Up_Day'].shift(1)).cumsum()).cumcount() + 1
        df_seq['Down_Streak'] = df_seq['Up_Day'].replace({1: 0, 0: 1}).groupby((df_seq['Up_Day'] != df_seq['Up_Day'].shift(1)).cumsum()).cumcount() + 1
        
        # Calculate mean reversion potential
        if 'SMA_50' in df_seq.columns:
            df_seq['Dist_From_SMA50_Pct'] = (df_seq['Close'] - df_seq['SMA_50']) / df_seq['SMA_50']
        
        if 'RSI_14' in df_seq.columns:
            df_seq['Overbought'] = (df_seq['RSI_14'] > 70).astype(int)
            df_seq['Oversold'] = (df_seq['RSI_14'] < 30).astype(int)
        
        # Create a composite signal feature if all necessary components exist
        if all(col in df_seq.columns for col in ['Uptrend', 'Momentum_Regime', 'Overbought', 'Oversold']):
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