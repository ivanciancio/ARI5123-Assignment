"""
Feature Engineering Module

This module handles feature preparation for the CNN trading model.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for CNN trading model."""
    
    def __init__(self):
        """Initialise the FeatureEngineer."""
        pass
    
    def create_bar_chart_image_improved(self, price_data, window_size=30):
        """
        Create 30x30 binary image with improved normalisation.
        """
        # Create empty image
        img = np.zeros((30, 30), dtype=np.float32)
        
        # Robust normalisation using percentiles
        p5 = np.percentile(price_data, 5)
        p95 = np.percentile(price_data, 95)
        price_range = p95 - p5
        
        if price_range > 0:
            for i in range(min(window_size, len(price_data))):
                # Clip extreme values and normalise
                clipped_price = np.clip(price_data[i], p5, p95)
                normalised_price = int(29 * (clipped_price - p5) / price_range)
                normalised_price = np.clip(normalised_price, 0, 29)
                
                # Draw bar
                img[29-normalised_price:30, i] = 1.0
        
        # Add small noise for generalisation
        noise = np.random.normal(0, 0.01, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        return img
    
    def generate_improved_labels(self, data, ticker, window_size=30, 
                            prediction_horizon=5, train_data=None):
        """
        Generate labels with adaptive thresholds.
        """
        close_prices = data['Close'].values
        
        # Calculate volatility for adaptive thresholds
        returns = pd.Series(close_prices).pct_change().dropna()
        volatility = returns.std()
        
        # Set adaptive thresholds
        if train_data is not None:
            train_prices = train_data['Close'].values
            train_returns = pd.Series(train_prices).pct_change().dropna()
            reference_vol = train_returns.std()
            
            base_threshold = reference_vol * 0.3
            threshold_high = base_threshold
            threshold_low = -base_threshold
        else:
            # Fallback to fixed thresholds
            threshold_high = volatility * 0.3
            threshold_low = -volatility * 0.3
        
        # Generate labels
        labels = []
        
        for i in range(len(close_prices) - window_size - prediction_horizon):
            current_price = close_prices[i + window_size - 1]
            future_price = close_prices[i + window_size + prediction_horizon - 1]
            
            if current_price > 0:
                price_return = (future_price - current_price) / current_price
                
                # Classify
                if price_return > threshold_high:
                    label = 1  # Buy
                elif price_return < threshold_low:
                    label = 2  # Sell
                else:
                    label = 0  # Hold
            else:
                label = 0  # Hold for invalid prices
            
            labels.append(label)
        
        return np.array(labels)
    
    def balance_dataset(self, X, y, method='undersample'):
        """
        Balance the dataset to prevent bias towards majority class.
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        
        if len(unique_labels) < 2:
            return X, y
        
        if method == 'undersample':
            # Undersample to the minority class
            min_count = np.min(counts)
            
            balanced_X = []
            balanced_y = []
            
            for label in unique_labels:
                mask = (y == label)
                X_class = X[mask]
                y_class = y[mask]
                
                if len(X_class) > min_count:
                    indices = np.random.choice(len(X_class), min_count, replace=False)
                    X_class = X_class[indices]
                    y_class = y_class[indices]
                
                balanced_X.append(X_class)
                balanced_y.append(y_class)
            
            X_balanced = np.vstack(balanced_X)
            y_balanced = np.hstack(balanced_y)
            
        elif method == 'oversample':
            # Oversample to the majority class
            max_count = np.max(counts)
            
            balanced_X = []
            balanced_y = []
            
            for label in unique_labels:
                mask = (y == label)
                X_class = X[mask]
                y_class = y[mask]
                
                if len(X_class) < max_count:
                    indices = np.random.choice(len(X_class), max_count, replace=True)
                    X_class = X_class[indices]
                    y_class = y_class[indices]
                
                balanced_X.append(X_class)
                balanced_y.append(y_class)
            
            X_balanced = np.vstack(balanced_X)
            y_balanced = np.hstack(balanced_y)
            
        else:  # hybrid
            # Target middle ground
            target_count = int(np.mean(counts))
            
            balanced_X = []
            balanced_y = []
            
            for label in unique_labels:
                mask = (y == label)
                X_class = X[mask]
                y_class = y[mask]
                
                if len(X_class) > target_count:
                    indices = np.random.choice(len(X_class), target_count, replace=False)
                    X_class = X_class[indices]
                    y_class = y_class[indices]
                elif len(X_class) < target_count:
                    indices = np.random.choice(len(X_class), target_count, replace=True)
                    X_class = X_class[indices]
                    y_class = y_class[indices]
                
                balanced_X.append(X_class)
                balanced_y.append(y_class)
            
            X_balanced = np.vstack(balanced_X)
            y_balanced = np.hstack(balanced_y)
        
        # Shuffle the balanced dataset
        shuffle_indices = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_indices]
        y_balanced = y_balanced[shuffle_indices]
        
        return X_balanced, y_balanced
    
    def prepare_benchmark_features(self, train_data, test_data, ticker, window_size=30):
        """
        Prepare features following benchmark methodology.
        """
        # Process training data
        train_images = []
        train_close_prices = train_data['Close'].values
        
        for i in range(len(train_close_prices) - window_size - 5):  # 5-day prediction
            price_window = train_close_prices[i:i + window_size]
            img = self.create_bar_chart_image_improved(price_window, window_size)
            train_images.append(img)
        
        # Generate labels
        train_labels = self.generate_improved_labels(
            train_data, ticker, window_size, 5, train_data
        )
        
        # Process testing data
        test_images = []
        test_close_prices = test_data['Close'].values
        
        for i in range(len(test_close_prices) - window_size - 5):
            price_window = test_close_prices[i:i + window_size]
            img = self.create_bar_chart_image_improved(price_window, window_size)
            test_images.append(img)
        
        test_labels = self.generate_improved_labels(
            test_data, ticker, window_size, 5, train_data
        )
        
        # Convert to numpy arrays
        X_train = np.array(train_images, dtype=np.float32)
        y_train = np.array(train_labels, dtype=np.int64)
        X_test = np.array(test_images, dtype=np.float32)
        y_test = np.array(test_labels, dtype=np.int64)
        
        # Ensure matching lengths
        min_train_len = min(len(X_train), len(y_train))
        min_test_len = min(len(X_test), len(y_test))
        
        X_train = X_train[:min_train_len]
        y_train = y_train[:min_train_len]
        X_test = X_test[:min_test_len]
        y_test = y_test[:min_test_len]
        
        # Balance training dataset
        X_train_balanced, y_train_balanced = self.balance_dataset(
            X_train, y_train, method='hybrid'
        )
        
        # Create validation split
        val_split = int(0.8 * len(X_train_balanced))
        
        # Shuffle before splitting
        shuffle_indices = np.random.permutation(len(X_train_balanced))
        X_train_balanced = X_train_balanced[shuffle_indices]
        y_train_balanced = y_train_balanced[shuffle_indices]
        
        X_val = X_train_balanced[val_split:]
        y_val = y_train_balanced[val_split:]
        X_train_final = X_train_balanced[:val_split]
        y_train_final = y_train_balanced[:val_split]
        
        return {
            'X_train': X_train_final,
            'y_train': y_train_final,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'ticker': ticker,
            'window_size': window_size,
            'prediction_horizon': 5,
        }