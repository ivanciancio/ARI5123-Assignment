"""
Trading Strategy Module

This module implements the trading strategy based on model predictions.
It follows the approach from Sezer & Ozbayoglu (2018) for buy/sell signals.
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
    Follows the approach from Sezer & Ozbayoglu (2018) paper.
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
        self.entry_price = 0
        self.entry_cost = 0
    
    def apply_strategy(self, prices, signals, dates=None, fixed_threshold=None):
        """
        Apply the trading strategy based on model signals.
        Implementation follows Sezer & Ozbayoglu (2018) paper.
        
        Args:
            prices: Array of prices
            signals: Array of trading signals (0=Hold, 1=Buy, 2=Sell)
            dates: Array of dates for the trading period (optional)
            fixed_threshold: Fixed threshold value (if None, use dynamic thresholds)
            
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
        
        # Track trading activity
        buy_signals = 0
        sell_signals = 0
        
        for i in range(1, len(prices)):
            # Current portfolio value
            current_value = self.capital
            if self.position > 0:
                current_value += self.position * prices[i-1]
            
            # Generate trading decision from signal
            # If signals are probabilities, convert to class (0=Hold, 1=Buy, 2=Sell)
            if isinstance(signals[i-1], (np.ndarray, list)) and len(np.shape(signals)) > 1:
                signal_class = np.argmax(signals[i-1])
            else:
                signal_class = int(signals[i-1])
            
            # Execute trades based on decision
            if signal_class == 1 and self.position == 0:
                # BUY
                # Calculate position size (invest 95% of capital)
                amount_to_invest = self.capital * 0.95
                cost = amount_to_invest * self.transaction_cost
                shares = (amount_to_invest - cost) / prices[i]
                
                self.position = shares
                self.capital -= (amount_to_invest)
                self.entry_price = prices[i]  # Record entry price for PnL calculation
                self.entry_cost = cost  # Record entry cost
                
                # Record trade
                self.trades.append({
                    'date': dates[i],
                    'type': 'BUY',
                    'price': prices[i],
                    'shares': shares,
                    'value': shares * prices[i],
                    'cost': cost,
                    'pnl': 0  # PnL is 0 for buy trades
                })
                
                buy_signals += 1
                logger.info(f"BUY: {dates[i]}, Price: {prices[i]:.2f}, Shares: {shares:.2f}, Capital: {self.capital:.2f}")
                
            elif signal_class == 2 and self.position > 0:
                # SELL
                # Calculate trade value and cost
                value = self.position * prices[i]
                cost = value * self.transaction_cost
                
                # Calculate PnL for this trade
                entry_value = self.position * self.entry_price
                exit_value = value - cost
                pnl = exit_value - entry_value - self.entry_cost
                
                # Update capital
                self.capital += value - cost
                
                # Record trade
                self.trades.append({
                    'date': dates[i],
                    'type': 'SELL',
                    'price': prices[i],
                    'shares': self.position,
                    'value': value,
                    'cost': cost,
                    'pnl': pnl
                })
                
                logger.info(f"SELL: {dates[i]}, Price: {prices[i]:.2f}, Shares: {self.position:.2f}, PnL: {pnl:.2f}, Capital: {self.capital:.2f}")
                
                # Reset position
                self.position = 0
                sell_signals += 1
            
            # Calculate current portfolio value
            portfolio_value = self.capital
            if self.position > 0:
                portfolio_value += self.position * prices[i]
            
            self.portfolio_values.append(portfolio_value)
            
            # Calculate daily return
            if i > 0:
                daily_return = (portfolio_value - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
                self.returns.append(daily_return)
        
        # Close any remaining position at the end of the period
        if self.position > 0:
            final_price = prices[-1]
            value = self.position * final_price
            cost = value * self.transaction_cost
            
            # Calculate PnL for final trade
            entry_value = self.position * self.entry_price
            exit_value = value - cost
            pnl = exit_value - entry_value - self.entry_cost
            
            # Record trade
            self.trades.append({
                'date': dates[-1] if dates is not None else len(prices) - 1,
                'type': 'SELL_FINAL',
                'price': final_price,
                'shares': self.position,
                'value': value,
                'cost': cost,
                'pnl': pnl
            })
            
            # Update capital
            self.capital += value - cost
            
            # Update final portfolio value
            self.portfolio_values[-1] = self.capital
            
            # Reset position
            self.position = 0
            sell_signals += 1
            
            logger.info(f"FINAL SELL: {dates[-1]}, Price: {final_price:.2f}, Shares: {self.position:.2f}, PnL: {pnl:.2f}, Capital: {self.capital:.2f}")
        
        # Print statistics for debugging
        logger.info(f"Strategy statistics: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
        logger.info(f"Final portfolio value: {self.portfolio_values[-1]:.2f}")
        
        # Return performance metrics
        metrics = self._calculate_performance_metrics()
        logger.info(f"Performance metrics: {metrics}")
        return metrics
    
    def apply_buy_and_hold(self, prices, dates=None):
        """
        Apply a simple buy and hold strategy as a benchmark.
        
        Args:
            prices: Array of prices
            dates: Array of dates for the trading period (optional)
            
        Returns:
            Dictionary with strategy performance metrics
        """
        self.reset()
        
        # Use dates if provided, otherwise use indices
        if dates is None:
            dates = np.arange(len(prices))
        
        # Initial portfolio value
        self.portfolio_values.append(self.capital)
        
        # Buy at the beginning
        if len(prices) > 1:
            # Calculate position size (invest 95% of capital)
            amount_to_invest = self.capital * 0.95
            cost = amount_to_invest * self.transaction_cost
            shares = (amount_to_invest - cost) / prices[0]
            
            self.position = shares
            self.capital -= (amount_to_invest)
            self.entry_price = prices[0]  # Record entry price for PnL calculation
            self.entry_cost = cost  # Record entry cost
            
            # Record trade
            self.trades.append({
                'date': dates[0],
                'type': 'BUY',
                'price': prices[0],
                'shares': shares,
                'value': shares * prices[0],
                'cost': cost,
                'pnl': 0  # PnL is 0 for buy trades
            })
            
            # Calculate portfolio values for each day
            for i in range(1, len(prices)):
                # Portfolio value = cash + position value
                portfolio_value = self.capital + (self.position * prices[i])
                self.portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                daily_return = (portfolio_value - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
                self.returns.append(daily_return)
            
            # Sell at the end
            final_price = prices[-1]
            value = self.position * final_price
            cost = value * self.transaction_cost
            
            # Calculate PnL for this trade
            entry_value = self.position * self.entry_price
            exit_value = value - cost
            pnl = exit_value - entry_value - self.entry_cost
            
            # Record trade
            self.trades.append({
                'date': dates[-1],
                'type': 'SELL_FINAL',
                'price': final_price,
                'shares': self.position,
                'value': value,
                'cost': cost,
                'pnl': pnl
            })
            
            # Update capital
            self.capital += value - cost
            
            # Update final portfolio value
            self.portfolio_values[-1] = self.capital
            
            # Reset position
            self.position = 0
        
        # Return performance metrics
        metrics = self._calculate_performance_metrics()
        logger.info(f"Buy & Hold metrics: {metrics}")
        return metrics
    
    def apply_random_strategy(self, prices, dates=None, seed=42):
        """
        Apply a random trading strategy as a benchmark.
        
        Args:
            prices: Array of prices
            dates: Array of dates for the trading period (optional)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with strategy performance metrics
        """
        self.reset()
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Use dates if provided, otherwise use indices
        if dates is None:
            dates = np.arange(len(prices))
        
        # Initial portfolio value
        self.portfolio_values.append(self.capital)
        
        # Generate random signals between 0 and 1
        signals = np.random.random(len(prices))
        
        # Apply strategy with random signals
        for i in range(1, len(prices)):
            # Current portfolio value
            current_value = self.capital
            if self.position > 0:
                current_value += self.position * prices[i-1]
            
            # Random signal between 0 and 1
            signal = signals[i-1]
            
            # Execute trades based on signal (50% threshold)
            if signal > 0.5 and self.position == 0:
                # BUY
                # Calculate position size (invest 95% of capital)
                amount_to_invest = self.capital * 0.95
                cost = amount_to_invest * self.transaction_cost
                shares = (amount_to_invest - cost) / prices[i]
                
                self.position = shares
                self.capital -= (amount_to_invest)
                self.entry_price = prices[i]  # Record entry price for PnL calculation
                self.entry_cost = cost  # Record entry cost
                
                # Record trade
                self.trades.append({
                    'date': dates[i],
                    'type': 'BUY',
                    'price': prices[i],
                    'shares': shares,
                    'value': shares * prices[i],
                    'cost': cost,
                    'pnl': 0  # PnL is 0 for buy trades
                })
                
            elif signal <= 0.5 and self.position > 0:
                # SELL
                # Calculate trade value and cost
                value = self.position * prices[i]
                cost = value * self.transaction_cost
                
                # Calculate PnL for this trade
                entry_value = self.position * self.entry_price
                exit_value = value - cost
                pnl = exit_value - entry_value - self.entry_cost
                
                # Update capital
                self.capital += value - cost
                
                # Record trade
                self.trades.append({
                    'date': dates[i],
                    'type': 'SELL',
                    'price': prices[i],
                    'shares': self.position,
                    'value': value,
                    'cost': cost,
                    'pnl': pnl
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
        
        # Close any remaining position at the end of the period
        if self.position > 0:
            final_price = prices[-1]
            value = self.position * final_price
            cost = value * self.transaction_cost
            
            # Calculate PnL for this trade
            entry_value = self.position * self.entry_price
            exit_value = value - cost
            pnl = exit_value - entry_value - self.entry_cost
            
            # Record trade
            self.trades.append({
                'date': dates[-1] if dates is not None else len(prices) - 1,
                'type': 'SELL_FINAL',
                'price': final_price,
                'shares': self.position,
                'value': value,
                'cost': cost,
                'pnl': pnl
            })
            
            # Update capital
            self.capital += value - cost
            
            # Update final portfolio value
            self.portfolio_values[-1] = self.capital
            
            # Reset position
            self.position = 0
        
        # Return performance metrics
        metrics = self._calculate_performance_metrics()
        logger.info(f"Random strategy metrics: {metrics}")
        return metrics
    
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics for the trading strategy.
        
        Returns:
            Dictionary with performance metrics
        """
        # Convert returns to numpy array and handle empty case
        returns = np.array(self.returns) if self.returns else np.array([0.0])
        
        # Calculate total return
        if len(self.portfolio_values) >= 2:
            total_return = (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital
        else:
            total_return = 0.0
        
        # Calculate annualised return (assuming 252 trading days per year)
        n_days = len(returns) if len(returns) > 0 else 1
        annualized_return = ((1 + total_return) ** (252 / n_days) - 1) if total_return > -1 else -1
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns_std = np.std(returns)
        sharpe_ratio = np.mean(returns) * 252 / (returns_std * np.sqrt(252)) if returns_std > 0 else 0
        
        # Calculate maximum drawdown
        max_drawdown = 0.0
        if len(self.portfolio_values) > 1:
            portfolio_values = np.array(self.portfolio_values)
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Extract sell trades for win rate calculation
        sell_trades = [t for t in self.trades if t['type'] in ('SELL', 'SELL_FINAL')]
        
        # Default values
        win_rate = 0.0
        profit_factor = 1.0
        num_trades = len(sell_trades)
        
        # Calculate win rate and profit factor if there are any trades
        if sell_trades:
            # Calculate PnL for each sell trade if not already present
            for trade in sell_trades:
                if 'pnl' not in trade:
                    # This shouldn't happen with our new implementation, but just in case
                    trade['pnl'] = 0.0
            
            # Count winning trades
            winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
            
            # Calculate profit factor
            total_profits = sum(max(0, t.get('pnl', 0)) for t in sell_trades)
            total_losses = sum(abs(min(0, t.get('pnl', 0))) for t in sell_trades)
            profit_factor = total_profits / total_losses if total_losses > 0 else 1
        
        # Return metrics dictionary with non-zero minimum values
        return {
            'total_return': max(total_return, 0.00001) if total_return > 0 else min(total_return, -0.00001),
            'annualized_return': max(annualized_return, 0.00001) if annualized_return > 0 else min(annualized_return, -0.00001),
            'sharpe_ratio': max(sharpe_ratio, 0.001) if sharpe_ratio > 0 else min(sharpe_ratio, -0.001),
            'max_drawdown': max(max_drawdown, 0.00001),
            'win_rate': max(win_rate, 0.00001) if win_rate > 0 else 0,
            'profit_factor': max(profit_factor, 0.001),
            'number_of_trades': num_trades,
            'final_value': max(self.portfolio_values[-1], 0.01) if self.portfolio_values else self.initial_capital
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
        return np.array(self.returns) if self.returns else np.array([0.0])
    
    def get_trades(self):
        """
        Get the trades executed by the strategy.
        
        Returns:
            DataFrame of trades
        """
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()