"""
Trading Strategy Module - Fixed Version

Key Changes:
1. Fixed return calculation inconsistencies
2. Improved threshold selection (use fixed by default)
3. Better position sizing
4. More conservative trading approach
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Fixed trading strategy with corrected return calculations and better thresholds.
    """
    
    def __init__(self, initial_capital=10000.0, transaction_cost=0.001, max_position_size=0.95):
        """
        Initialize trading strategy with conservative defaults.
        
        Args:
            initial_capital: Initial capital for the portfolio
            transaction_cost: Transaction cost as percentage (0.001 = 0.1%)
            max_position_size: Maximum portion of capital to invest (0.95 = 95%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
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
    
    def apply_strategy(self, prices, signals, dates=None, fixed_threshold=0.25):
        """
        Apply trading strategy with FIXED threshold by default (performs better).
        
        Key improvements:
        - Use fixed threshold 0.25 instead of dynamic (based on your results)
        - More conservative position sizing
        - Better signal interpretation
        - Fixed return calculations
        """
        if len(prices) != len(signals):
            raise ValueError("Prices and signals must have the same length")

        self.reset()
        
        if dates is None:
            dates = np.arange(len(prices))
        
        self.portfolio_values.append(self.capital)
        
        logger.info(f"Using FIXED threshold: {fixed_threshold}")
        
        is_multiclass = len(signals.shape) > 1 and signals.shape[1] > 1
        
        # Use FIXED threshold (performs better based on your results)
        if fixed_threshold is None:
            # Fallback to dynamic if explicitly requested
            if is_multiclass:
                max_probs = np.max(signals, axis=1)
                confidence_threshold = np.percentile(max_probs, 60)
                logger.info(f"Dynamic confidence threshold: {confidence_threshold:.4f}")
            else:
                confidence_threshold = np.median(signals)
        else:
            confidence_threshold = fixed_threshold
            logger.info(f"Using FIXED threshold: {confidence_threshold}")
        
        buy_signals = 0
        sell_signals = 0
        trades_executed = 0
        
        for i in range(1, len(prices)):
            if is_multiclass:
                probs = signals[i-1]  # [Hold, Buy, Sell] probabilities
                
                if fixed_threshold is not None:
                    # FIXED threshold method (better performance)
                    buy_prob = probs[1]   # Buy probability
                    sell_prob = probs[2]  # Sell probability
                    hold_prob = probs[0]  # Hold probability
                    
                    # More conservative: require clear signal superiority
                    if (buy_prob >= confidence_threshold and 
                        buy_prob > sell_prob + 0.1 and
                        buy_prob > hold_prob):
                        signal_class = 1  # Buy
                    elif (sell_prob >= confidence_threshold and 
                          sell_prob > buy_prob + 0.1 and
                          sell_prob > hold_prob):
                        signal_class = 2  # Sell
                    else:
                        signal_class = 0  # Hold
                else:
                    # Dynamic method (fallback)
                    max_prob = np.max(probs)
                    winning_class = np.argmax(probs)
                    
                    if max_prob > confidence_threshold:
                        signal_class = winning_class
                    else:
                        signal_class = 0  # Hold
            else:
                # Single value signal
                signal_value = signals[i-1] if not hasattr(signals[i-1], '__len__') else signals[i-1][0]
                signal_class = 1 if signal_value >= confidence_threshold else 0
            
            # Execute trades with improved logic
            if signal_class == 1 and self.position == 0:
                # BUY - More conservative position sizing
                amount_to_invest = self.capital * self.max_position_size
                cost = amount_to_invest * self.transaction_cost
                shares = (amount_to_invest - cost) / prices[i]
                
                if shares > 0:  # Ensure valid trade
                    self.position = shares
                    self.capital -= amount_to_invest
                    self.entry_price = prices[i]
                    self.entry_cost = cost
                    
                    self.trades.append({
                        'date': dates[i],
                        'type': 'BUY',
                        'price': prices[i],
                        'shares': shares,
                        'value': shares * prices[i],
                        'cost': cost,
                        'pnl': 0,
                        'capital_after': self.capital
                    })
                    
                    buy_signals += 1
                    trades_executed += 1
                    
                    if trades_executed <= 5:  # Log first few trades
                        logger.info(f"BUY at {prices[i]:.2f}, Shares: {shares:.2f}")
                
            elif signal_class == 2 and self.position > 0:
                # SELL
                value = self.position * prices[i]
                cost = value * self.transaction_cost
                
                # Calculate P&L correctly
                entry_value = self.position * self.entry_price
                exit_value = value - cost
                pnl = exit_value - entry_value - self.entry_cost
                
                self.capital += value - cost
                
                self.trades.append({
                    'date': dates[i],
                    'type': 'SELL',
                    'price': prices[i],
                    'shares': self.position,
                    'value': value,
                    'cost': cost,
                    'pnl': pnl,
                    'capital_after': self.capital
                })
                
                self.position = 0
                sell_signals += 1
                trades_executed += 1
                
                if trades_executed <= 5:  # Log first few trades
                    logger.info(f"SELL at {prices[i]:.2f}, PnL: {pnl:.2f}")
            
            # Calculate portfolio value CORRECTLY
            portfolio_value = self.capital
            if self.position > 0:
                portfolio_value += self.position * prices[i]
            
            self.portfolio_values.append(portfolio_value)
            
            # Calculate daily return correctly
            if i > 0:
                daily_return = (portfolio_value - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
                self.returns.append(daily_return)
        
        # Close final position if needed
        if self.position > 0:
            final_price = prices[-1]
            value = self.position * final_price
            cost = value * self.transaction_cost
            
            entry_value = self.position * self.entry_price
            exit_value = value - cost
            pnl = exit_value - entry_value - self.entry_cost
            
            self.trades.append({
                'date': dates[-1] if dates is not None else len(prices) - 1,
                'type': 'SELL_FINAL',
                'price': final_price,
                'shares': self.position,
                'value': value,
                'cost': cost,
                'pnl': pnl,
                'capital_after': self.capital + value - cost
            })
            
            self.capital += value - cost
            self.portfolio_values[-1] = self.capital
            self.position = 0
        
        logger.info(f"Strategy executed: Buy={buy_signals}, Sell={sell_signals}, Total trades={trades_executed}")
        logger.info(f"Final portfolio value: {self.portfolio_values[-1]:.2f}")
        
        return self._calculate_performance_metrics_fixed()
    
    def apply_buy_and_hold(self, prices, dates=None):
        """Buy and hold strategy with FIXED return calculations."""
        self.reset()
        
        if dates is None:
            dates = np.arange(len(prices))
        
        self.portfolio_values.append(self.capital)
        
        if len(prices) > 1:
            # Buy at the beginning
            amount_to_invest = self.capital * self.max_position_size
            cost = amount_to_invest * self.transaction_cost
            shares = (amount_to_invest - cost) / prices[0]
            
            self.position = shares
            self.capital -= amount_to_invest
            self.entry_price = prices[0]
            self.entry_cost = cost
            
            # Record buy trade
            self.trades.append({
                'date': dates[0],
                'type': 'BUY',
                'price': prices[0],
                'shares': shares,
                'value': shares * prices[0],
                'cost': cost,
                'pnl': 0,
                'capital_after': self.capital
            })
            
            # Calculate portfolio values for each day
            for i in range(1, len(prices)):
                portfolio_value = self.capital + (self.position * prices[i])
                self.portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                daily_return = (portfolio_value - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
                self.returns.append(daily_return)
            
            # Sell at the end
            final_price = prices[-1]
            value = self.position * final_price
            cost = value * self.transaction_cost
            
            # Calculate P&L
            entry_value = self.position * self.entry_price
            exit_value = value - cost
            pnl = exit_value - entry_value - self.entry_cost
            
            # Record sell trade
            self.trades.append({
                'date': dates[-1],
                'type': 'SELL_FINAL',
                'price': final_price,
                'shares': self.position,
                'value': value,
                'cost': cost,
                'pnl': pnl,
                'capital_after': self.capital + value - cost
            })
            
            # Update final values
            self.capital += value - cost
            self.portfolio_values[-1] = self.capital
            self.position = 0
        
        return self._calculate_performance_metrics_fixed()
    
    def apply_random_strategy(self, prices, dates=None, seed=42):
        """Random strategy with same fixed calculations."""
        self.reset()
        np.random.seed(seed)
        
        if dates is None:
            dates = np.arange(len(prices))
        
        self.portfolio_values.append(self.capital)
        signals = np.random.random(len(prices))
        
        for i in range(1, len(prices)):
            signal = signals[i-1]
            
            if signal > 0.6 and self.position == 0:  # Higher threshold for random
                # BUY
                amount_to_invest = self.capital * self.max_position_size
                cost = amount_to_invest * self.transaction_cost
                shares = (amount_to_invest - cost) / prices[i]
                
                if shares > 0:
                    self.position = shares
                    self.capital -= amount_to_invest
                    self.entry_price = prices[i]
                    self.entry_cost = cost
                    
                    self.trades.append({
                        'date': dates[i],
                        'type': 'BUY',
                        'price': prices[i],
                        'shares': shares,
                        'value': shares * prices[i],
                        'cost': cost,
                        'pnl': 0,
                        'capital_after': self.capital
                    })
                
            elif signal <= 0.4 and self.position > 0:  # Lower threshold for random
                # SELL
                value = self.position * prices[i]
                cost = value * self.transaction_cost
                
                entry_value = self.position * self.entry_price
                exit_value = value - cost
                pnl = exit_value - entry_value - self.entry_cost
                
                self.capital += value - cost
                
                self.trades.append({
                    'date': dates[i],
                    'type': 'SELL',
                    'price': prices[i],
                    'shares': self.position,
                    'value': value,
                    'cost': cost,
                    'pnl': pnl,
                    'capital_after': self.capital
                })
                
                self.position = 0
            
            # Calculate portfolio value
            portfolio_value = self.capital
            if self.position > 0:
                portfolio_value += self.position * prices[i]
            
            self.portfolio_values.append(portfolio_value)
            
            if i > 0:
                daily_return = (portfolio_value - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
                self.returns.append(daily_return)
        
        # Close final position
        if self.position > 0:
            final_price = prices[-1]
            value = self.position * final_price
            cost = value * self.transaction_cost
            
            entry_value = self.position * self.entry_price
            exit_value = value - cost
            pnl = exit_value - entry_value - self.entry_cost
            
            self.trades.append({
                'date': dates[-1],
                'type': 'SELL_FINAL',
                'price': final_price,
                'shares': self.position,
                'value': value,
                'cost': cost,
                'pnl': pnl,
                'capital_after': self.capital + value - cost
            })
            
            self.capital += value - cost
            self.portfolio_values[-1] = self.capital
            self.position = 0
        
        return self._calculate_performance_metrics_fixed()
    
    def _calculate_performance_metrics_fixed(self):
        """
        FIXED performance metrics calculation to resolve inconsistencies.
        """
        returns = np.array(self.returns) if self.returns else np.array([0.0])
        
        # FIXED: Calculate total return correctly
        if len(self.portfolio_values) >= 2:
            total_return = (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital
        else:
            total_return = 0.0
        
        # FIXED: Calculate annualized return correctly
        n_days = len(returns) if len(returns) > 0 else 1
        years = n_days / 252.0  # Trading days per year
        
        # Correct annualized return formula
        if total_return > -1 and years > 0:
            annualised_return = ((1 + total_return) ** (1/years)) - 1
        else:
            annualised_return = -1
        
        # FIXED: Sharpe ratio calculation
        if len(returns) > 1:
            returns_std = np.std(returns, ddof=1)  # Sample standard deviation
            if returns_std > 0:
                sharpe_ratio = (np.mean(returns) * 252) / (returns_std * np.sqrt(252))
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # FIXED: Maximum drawdown calculation
        max_drawdown = 0.0
        if len(self.portfolio_values) > 1:
            portfolio_values = np.array(self.portfolio_values)
            # Calculate running maximum
            peak = np.maximum.accumulate(portfolio_values)
            # Calculate drawdown as percentage
            drawdown = (peak - portfolio_values) / peak
            # Handle division by zero
            drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
            max_drawdown = np.max(drawdown)
        
        # Trading metrics
        sell_trades = [t for t in self.trades if t['type'] in ('SELL', 'SELL_FINAL')]
        
        win_rate = 0.0
        profit_factor = 1.0
        num_trades = len(sell_trades)
        
        if sell_trades:
            # Ensure PnL is calculated
            for trade in sell_trades:
                if 'pnl' not in trade:
                    trade['pnl'] = 0.0
            
            # Win rate
            winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
            
            # Profit factor
            total_profits = sum(max(0, t.get('pnl', 0)) for t in sell_trades)
            total_losses = sum(abs(min(0, t.get('pnl', 0))) for t in sell_trades)
            profit_factor = total_profits / total_losses if total_losses > 0 else (total_profits if total_profits > 0 else 1.0)
        
        return {
            # Core metrics
            'total_return': total_return,
            'annualised_return': annualised_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'number_of_trades': num_trades,
            'final_value': self.portfolio_values[-1] if self.portfolio_values else self.initial_capital,
            
            # Additional metrics
            'total_trades': len(self.trades),
            'buy_trades': len([t for t in self.trades if t['type'] == 'BUY']),
            'sell_trades': len(sell_trades),
            'years': years,
            
            # Debug info
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_pnl': sum(t.get('pnl', 0) for t in sell_trades)
        }
    
    def get_portfolio_values(self):
        """Get the portfolio values over time."""
        return np.array(self.portfolio_values)
    
    def get_returns(self):
        """Get the daily returns."""
        return np.array(self.returns) if self.returns else np.array([0.0])
    
    def get_trades(self):
        """Get the trades executed by the strategy."""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()