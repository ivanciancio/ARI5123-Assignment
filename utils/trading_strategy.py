"""
Trading Strategy Module

This module implements the trading strategy based on model predictions.
It includes position sizing, risk management, and performance evaluation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Trading strategy implementation based on CNN model predictions.
    """
    
    def __init__(self, initial_capital=10000.0, transaction_cost=0.001):
        """
        Initialise the trading strategy.
        
        Args:
            initial_capital: Initial capital for the portfolio
            transaction_cost: Transaction cost as a percentage of the trade value
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self):
        """Reset the strategy's state."""
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        self.returns = []
    
    def apply_buy_and_hold(self, prices):
        """
        Apply a simple buy and hold strategy.
        
        Args:
            prices: Array of prices
            
        Returns:
            Dictionary with strategy performance metrics
        """
        self.reset()
        
        # Buy at the first price
        shares = self.capital / prices[0]
        cost = self.capital * self.transaction_cost
        self.capital -= cost
        shares_value = shares * prices[0]
        
        # Record initial portfolio value
        initial_value = self.capital + shares_value
        self.portfolio_values.append(initial_value)
        
        # Hold until the end
        for i in range(1, len(prices)):
            shares_value = shares * prices[i]
            portfolio_value = self.capital + shares_value
            self.portfolio_values.append(portfolio_value)
            
            # Calculate daily return
            daily_return = (portfolio_value - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            self.returns.append(daily_return)
        
        # Calculate performance metrics
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        return self._calculate_performance_metrics()
    
    def apply_strategy(self, prices, signals, dates=None):
        """
        Apply the trading strategy based on model signals.
        
        Args:
            prices: Array of prices
            signals: Array of trading signals (probabilities from 0 to 1)
            dates: Array of dates for the trading period (optional)
            
        Returns:
            Dictionary with strategy performance metrics
        """
        if len(prices) != len(signals):
            raise ValueError("Prices and signals must have the same length")
        
        self.reset()
        
        # Use dates if provided, otherwise use indices
        if dates is None:
            dates = np.arange(len(prices))
        
        # Initial portfolio value
        self.portfolio_values.append(self.capital)
        
        for i in range(1, len(prices)):
            # Current portfolio value
            current_value = self.capital
            if self.position > 0:
                current_value += self.position * prices[i-1]
            
            # Generate trading decision from signal
            signal = signals[i-1]
            
            # Convert probability to decision (>0.5 means buy/hold, <=0.5 means sell/stay out)
            decision = 1 if signal > 0.5 else 0
            
            # Execute trades based on decision
            if decision == 1 and self.position == 0:
                # BUY
                # Calculate position size (invest 95% of capital)
                amount_to_invest = self.capital * 0.95
                cost = amount_to_invest * self.transaction_cost
                shares = (amount_to_invest - cost) / prices[i]
                
                self.position = shares
                self.capital -= (amount_to_invest)
                
                # Record trade
                self.trades.append({
                    'date': dates[i],
                    'type': 'BUY',
                    'price': prices[i],
                    'shares': shares,
                    'value': shares * prices[i],
                    'cost': cost
                })
                
            elif decision == 0 and self.position > 0:
                # SELL
                # Calculate trade value and cost
                value = self.position * prices[i]
                cost = value * self.transaction_cost
                
                # Update capital
                self.capital += value - cost
                
                # Record trade
                self.trades.append({
                    'date': dates[i],
                    'type': 'SELL',
                    'price': prices[i],
                    'shares': self.position,
                    'value': value,
                    'cost': cost
                })
                
                # Reset position
                self.position = 0
            
            # Calculate current portfolio value
            portfolio_value = self.capital
            if self.position > 0:
                portfolio_value += self.position * prices[i]
            
            self.portfolio_values.append(portfolio_value)
            
            # Calculate daily return
            daily_return = (portfolio_value - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            self.returns.append(daily_return)
        
        # Return performance metrics
        return self._calculate_performance_metrics()
    
    def apply_random_strategy(self, prices, seed=42, dates=None):
        """
        Apply a random trading strategy for benchmarking.
        
        Args:
            prices: Array of prices
            seed: Random seed for reproducibility
            dates: Array of dates for the trading period (optional)
            
        Returns:
            Dictionary with strategy performance metrics
        """
        np.random.seed(seed)
        
        # Generate random signals
        signals = np.random.random(len(prices))
        
        # Apply strategy with random signals
        return self.apply_strategy(prices, signals, dates)
    
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics for the trading strategy.
        
        Returns:
            Dictionary with performance metrics
        """
        # Convert returns to numpy array
        returns = np.array(self.returns)
        
        # Calculate total return
        total_return = (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate annualised return (assuming 252 trading days per year)
        annualized_return = ((1 + total_return) ** (252 / len(returns)) - 1) if len(returns) > 0 else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0
        
        # Calculate maximum drawdown
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate win rate
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            # Group trades by pairs (buy followed by sell)
            trades_df['trade_id'] = trades_df.index // 2
            
            # Calculate profit/loss for each trade
            profits = []
            for trade_id in trades_df['trade_id'].unique():
                trade_group = trades_df[trades_df['trade_id'] == trade_id]
                if len(trade_group) == 2:  # Complete trade (buy and sell)
                    buy = trade_group[trade_group['type'] == 'BUY'].iloc[0]
                    sell = trade_group[trade_group['type'] == 'SELL'].iloc[0]
                    profit = sell['value'] - sell['cost'] - buy['value'] - buy['cost']
                    profits.append(profit)
            
            # Calculate win rate
            win_rate = np.mean(np.array(profits) > 0) if len(profits) > 0 else 0
            
            # Calculate average profit/loss
            avg_profit = np.mean(profits) if len(profits) > 0 else 0
            
            # Calculate profit factor
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p <= 0]
            profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) != 0 else float('inf')
        else:
            win_rate = 0
            avg_profit = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'profit_factor': profit_factor,
            'number_of_trades': len(self.trades) // 2,
            'final_value': self.portfolio_values[-1]
        }
    
    def get_portfolio_values(self):
        """
        Get the portfolio values over time.
        
        Returns:
            Array of portfolio values
        """
        return np.array(self.portfolio_values)
    
    def get_returns(self):
        """
        Get the daily returns.
        
        Returns:
            Array of daily returns
        """
        return np.array(self.returns)
    
    def get_trades(self):
        """
        Get the trades executed by the strategy.
        
        Returns:
            DataFrame of trades
        """
        return pd.DataFrame(self.trades)