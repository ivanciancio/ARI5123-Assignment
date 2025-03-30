"""
Visualisation Module

This module provides functions for visualising trading data, model performance, and strategy results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patheffects as PathEffects

# Set British English style for plots
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.style.use('seaborn-v0_8-darkgrid')


def plot_stock_data(data, ticker, start_date=None, end_date=None, figsize=(14, 8)):
    """
    Plot stock data with volume.
    
    Args:
        data: Dictionary of DataFrames with stock data
        ticker: Stock ticker to plot
        start_date: Start date for plotting
        end_date: End date for plotting
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if ticker not in data:
        raise ValueError(f"Ticker {ticker} not found in data")
    
    # Get stock data
    df = data[ticker].copy()
    
    # Filter by date if provided
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot price data
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['Open'], label='Open Price', color='green', alpha=0.5)
    
    # Add moving averages if available
    if 'SMA_20' in df.columns:
        ax1.plot(df.index, df['SMA_20'], label='20-day SMA', color='orange', linestyle='--')
    if 'SMA_50' in df.columns:
        ax1.plot(df.index, df['SMA_50'], label='50-day SMA', color='red', linestyle='--')
    
    # Customise price plot
    ax1.set_title(f"{ticker} Stock Price")
    ax1.set_ylabel("Price (£)")
    ax1.grid(True)
    ax1.legend()
    
    # Plot volume data
    ax2.bar(df.index, df['Volume'], color='purple', alpha=0.5)
    ax2.set_title(f"{ticker} Trading Volume")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.grid(True)
    
    # Format x-axis
    fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig


def plot_model_training_history(history, figsize=(12, 8)):
    """
    Plot model training history.
    
    Args:
        history: Training history (dict for PyTorch or History object for TensorFlow)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Check if history is a dict (PyTorch) or has history attribute (TensorFlow)
    history_dict = history if isinstance(history, dict) else history.history
    
    # Plot training and validation loss
    ax1.plot(history_dict['loss'], label='Training Loss')
    ax1.plot(history_dict['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot training and validation accuracy
    ax2.plot(history_dict['accuracy'], label='Training Accuracy')
    ax2.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def plot_model_evaluation(y_true, y_pred_prob, figsize=(16, 12)):
    """
    Plot model evaluation metrics including ROC curve and confusion matrix.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Receiver Operating Characteristic')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 1])
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_title('Confusion Matrix')
    
    # Plot precision-recall curve
    sorted_indices = np.argsort(y_pred_prob.flatten())
    thresholds = y_pred_prob.flatten()[sorted_indices]
    y_true_sorted = y_true[sorted_indices]
    
    precisions = []
    recalls = []
    
    for threshold in np.unique(thresholds):
        y_pred_threshold = (y_pred_prob >= threshold).astype(int).flatten()
        
        # Calculate precision and recall
        tp = np.sum((y_pred_threshold == 1) & (y_true == 1))
        fp = np.sum((y_pred_threshold == 1) & (y_true == 0))
        fn = np.sum((y_pred_threshold == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    axes[1, 0].plot(recalls, precisions)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].grid(True)
    
    # Plot signal distribution
    axes[1, 1].hist(y_pred_prob[y_true == 1], alpha=0.5, label='Positive class', bins=20)
    axes[1, 1].hist(y_pred_prob[y_true == 0], alpha=0.5, label='Negative class', bins=20)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Signal Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


def plot_portfolio_performance(portfolio_values, benchmark_values=None, dates=None, figsize=(14, 8)):
    """
    Plot portfolio performance compared to benchmark.
    
    Args:
        portfolio_values: Array of portfolio values
        benchmark_values: Array of benchmark values (optional)
        dates: Array of dates (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use dates if provided, otherwise use indices
    x = dates if dates is not None else np.arange(len(portfolio_values))
    
    # Plot portfolio values
    ax.plot(x, portfolio_values, label='Strategy', color='blue')
    
    # Plot benchmark values if provided
    if benchmark_values is not None:
        ax.plot(x, benchmark_values, label='Buy and Hold', color='green', linestyle='--')
    
    # Customise plot
    ax.set_title('Portfolio Performance')
    ax.set_ylabel('Portfolio Value (£)')
    ax.set_xlabel('Date' if dates is not None else 'Trading Day')
    ax.grid(True)
    ax.legend()
    
    # Format x-axis for dates
    if dates is not None:
        fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig


def plot_interactive_portfolio(portfolio_values, benchmark_values=None, dates=None, trades=None):
    """
    Create an interactive portfolio performance plot using Plotly.
    
    Args:
        portfolio_values: Array of portfolio values
        benchmark_values: Array of benchmark values (optional)
        dates: Array of dates (optional)
        trades: DataFrame of trade information (optional)
        
    Returns:
        Plotly figure
    """
    # Use dates if provided, otherwise use indices
    x = dates if dates is not None else np.arange(len(portfolio_values))
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add portfolio line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=portfolio_values,
            name="Strategy",
            line=dict(color='blue', width=2)
        )
    )
    
    # Add benchmark line if provided
    if benchmark_values is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=benchmark_values,
                name="Buy and Hold",
                line=dict(color='green', width=2, dash='dash')
            )
        )
    
    # Add trade markers if provided
    if trades is not None and len(trades) > 0:
        buy_trades = trades[trades['type'] == 'BUY']
        sell_trades = trades[trades['type'] == 'SELL']
        
        # Add buy markers
        if len(buy_trades) > 0:
            buy_y_values = []
            valid_buy_dates = []
            
            for date in buy_trades['date']:
                try:
                    if dates is not None:
                        # Try to find the date in the dates array
                        indices = np.where(dates == date)[0]
                        if len(indices) > 0:
                            buy_y_values.append(portfolio_values[indices[0]])
                            valid_buy_dates.append(date)
                        else:
                            # If date not found, try with closest date or skip
                            continue
                    else:
                        # If no dates provided, use integer index
                        idx = int(date)
                        if 0 <= idx < len(portfolio_values):
                            buy_y_values.append(portfolio_values[idx])
                            valid_buy_dates.append(date)
                except (ValueError, IndexError, TypeError):
                    # Skip this date if any error occurs
                    continue
            
            # Only add trace if we have valid points
            if len(buy_y_values) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=valid_buy_dates,
                        y=buy_y_values,
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color='green',
                            line=dict(color='green', width=2)
                        ),
                        name='Buy'
                    )
                )
        
        # Add sell markers
        if len(sell_trades) > 0:
            sell_y_values = []
            valid_sell_dates = []
            
            for date in sell_trades['date']:
                try:
                    if dates is not None:
                        # Try to find the date in the dates array
                        indices = np.where(dates == date)[0]
                        if len(indices) > 0:
                            sell_y_values.append(portfolio_values[indices[0]])
                            valid_sell_dates.append(date)
                        else:
                            # If date not found, try with closest date or skip
                            continue
                    else:
                        # If no dates provided, use integer index
                        idx = int(date)
                        if 0 <= idx < len(portfolio_values):
                            sell_y_values.append(portfolio_values[idx])
                            valid_sell_dates.append(date)
                except (ValueError, IndexError, TypeError):
                    # Skip this date if any error occurs
                    continue
            
            # Only add trace if we have valid points
            if len(sell_y_values) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=valid_sell_dates,
                        y=sell_y_values,
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color='red',
                            line=dict(color='red', width=2)
                        ),
                        name='Sell'
                    )
                )
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date" if dates is not None else "Trading Day",
        yaxis_title="Portfolio Value (£)",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        template="plotly_white"
    )
    
    # Format hover tooltip
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: £%{y:.2f}<extra></extra>"
    )
    
    return fig


def plot_strategy_comparison(strategies_results, figsize=(12, 10)):
    """
    Plot comparison of different trading strategies.
    
    Args:
        strategies_results: Dictionary of strategy names and their performance metrics
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract metrics for comparison
    metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    metrics_labels = ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    
    # Plot each metric
    for i, (metric, label) in enumerate(zip(metrics, metrics_labels)):
        values = [results[metric] for results in strategies_results.values()]
        
        # Handle percentage metrics
        if metric in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate']:
            values = [v * 100 for v in values]
            ylabel = f"{label} (%)"
        else:
            ylabel = label
        
        # Create bar chart
        bars = axes[i].bar(strategies_results.keys(), values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i].annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')
        
        # Customise subplot
        axes[i].set_title(label)
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True, axis='y')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, figsize=(10, 8)):
    """
    Plot feature importance for the model.
    Note: This works for models that provide feature_importances_ attribute.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot feature importance
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    
    # Customise plot
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.grid(True, axis='x')
    
    plt.tight_layout()
    return fig


def create_performance_report(strategy_results, benchmark_results=None):
    """
    Create a formatted performance report for the strategy.
    
    Args:
        strategy_results: Dictionary with strategy performance metrics
        benchmark_results: Dictionary with benchmark performance metrics (optional)
        
    Returns:
        Formatted HTML string with performance report
    """
    # Create HTML report
    html = """
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #2c3e50; text-align: center;">Trading Strategy Performance Report</h2>
        <div style="background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-top: 20px;">
            <h3 style="color: #3498db;">Performance Metrics</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #3498db; color: white;">
                    <th style="padding: 10px; text-align: left;">Metric</th>
                    <th style="padding: 10px; text-align: right;">Strategy</th>
    """
    
    # Add benchmark header if provided
    if benchmark_results:
        html += '<th style="padding: 10px; text-align: right;">Benchmark</th>'
    
    html += """
                </tr>
    """
    
    # Define metrics to display
    metrics = [
        ('total_return', 'Total Return', '{:.2%}'),
        ('annualized_return', 'Annualized Return', '{:.2%}'),
        ('sharpe_ratio', 'Sharpe Ratio', '{:.2f}'),
        ('max_drawdown', 'Maximum Drawdown', '{:.2%}'),
        ('win_rate', 'Win Rate', '{:.2%}'),
        ('profit_factor', 'Profit Factor', '{:.2f}'),
        ('number_of_trades', 'Number of Trades', '{:.0f}'),
        ('final_value', 'Final Portfolio Value', '£{:.2f}')
    ]
    
    # Add each metric to the table
    for i, (metric, label, fmt) in enumerate(metrics):
        # Determine row style
        row_style = 'background-color: #ecf0f1;' if i % 2 == 0 else ''
        
        # Format the values
        if metric == 'final_value':
            strategy_value = fmt.format(strategy_results[metric])
            if benchmark_results:
                benchmark_value = fmt.format(benchmark_results[metric])
        else:
            strategy_value = fmt.format(strategy_results[metric])
            if benchmark_results:
                benchmark_value = fmt.format(benchmark_results[metric])
        
        # Add row to the table
        html += f"""
            <tr style="{row_style}">
                <td style="padding: 10px; text-align: left;">{label}</td>
                <td style="padding: 10px; text-align: right;">{strategy_value}</td>
        """
        
        # Add benchmark value if provided
        if benchmark_results:
            # Determine if strategy outperformed benchmark
            if metric in ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 'final_value']:
                if strategy_results[metric] > benchmark_results[metric]:
                    cell_style = 'color: green;'
                elif strategy_results[metric] < benchmark_results[metric]:
                    cell_style = 'color: red;'
                else:
                    cell_style = ''
            elif metric in ['max_drawdown']:
                if strategy_results[metric] < benchmark_results[metric]:
                    cell_style = 'color: green;'
                elif strategy_results[metric] > benchmark_results[metric]:
                    cell_style = 'color: red;'
                else:
                    cell_style = ''
            else:
                cell_style = ''
                
            html += f"""
                <td style="padding: 10px; text-align: right; {cell_style}">{benchmark_value}</td>
            """
        
        html += """
            </tr>
        """
    
    # Close the table and add additional information
    html += """
            </table>
        </div>
        
        <div style="background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-top: 20px;">
            <h3 style="color: #3498db;">Interpretation</h3>
            <p style="line-height: 1.5;">
                The Sharpe ratio measures the risk-adjusted return, with higher values indicating better risk-adjusted performance.
                A Sharpe ratio above 1.0 is generally considered good, while above 2.0 is excellent.
            </p>
            <p style="line-height: 1.5;">
                The maximum drawdown represents the largest peak-to-trough decline and is a measure of downside risk.
                Lower values are better.
            </p>
            <p style="line-height: 1.5;">
                The win rate shows the percentage of profitable trades, while the profit factor is the ratio of gross profits to gross losses.
                A profit factor above 1.0 indicates a profitable strategy.
            </p>
        </div>
    </div>
    """
    
    return html  # Added the missing return statement


def create_improved_architecture_diagram():
    """
    Create a visually enhanced architecture diagram for the trading system
    with improved text clarity and proper spacing.
    
    Returns:
        Matplotlib figure
    """
    from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    
    # Create figure with a premium white background
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')  # Larger figure size
    ax.set_facecolor('white')
    
    # Remove axes
    ax.axis('off')
    
    # Define an improved color palette
    colors = {
        'primary': '#3b82f6',      # Main blue
        'secondary': '#1e40af',    # Dark blue
        'accent1': '#34d399',      # Teal accent
        'accent2': '#f97316',      # Orange accent
        'background': '#f8fafc',   # Very light gray
        'text_dark': '#0f172a',    # Near black
        'text_light': '#f8fafc',   # Near white
        'gradient_top': '#3b82f6', # Gradient top color
        'gradient_bottom': '#1e40af', # Gradient bottom color
        'arrow': '#0369a1',        # Arrow blue
        'shadow': (0, 0, 0, 0.5)   # Shadow color as a proper RGBA tuple
    }
    
    # Component details with improved descriptions
    components = [
        {
            'name': 'Historical\nStock Data',
            'desc': 'OHLCV data from\nYahoo Finance',
            'symbol': 'D',
            'color': '#4f46e5'  # Indigo
        },
        {
            'name': 'Data\nPreprocessing',
            'desc': 'Normalisation &\nsequence creation',
            'symbol': 'P',
            'color': '#8b5cf6'  # Purple
        },
        {
            'name': 'Feature\nEngineering',
            'desc': 'Technical indicators\n& patterns',
            'symbol': 'F',
            'color': '#3b82f6'  # Blue
        },
        {
            'name': 'CNN Model',
            'desc': 'Deep learning\nwith attention',
            'symbol': 'M',
            'color': '#0ea5e9'  # Sky blue
        },
        {
            'name': 'Trading\nSignals',
            'desc': 'Buy/sell probability\npredictions',
            'symbol': 'S',
            'color': '#14b8a6'  # Teal
        },
        {
            'name': 'Trading\nStrategy',
            'desc': 'Position sizing &\nrisk management',
            'symbol': 'T',
            'color': '#10b981'  # Emerald
        },
        {
            'name': 'Performance\nEvaluation',
            'desc': 'Return & risk\nmetrics analysis',
            'symbol': 'E',
            'color': '#22c55e'  # Green
        }
    ]
    
    # Add a premium background panel
    ax.add_patch(Rectangle(
        (0.03, 0.15), 0.94, 0.65, 
        facecolor=colors['background'],
        edgecolor='#e2e8f0',
        linewidth=1,
        alpha=0.8,
        zorder=0
    ))
    
    # Add a stylish header band
    ax.add_patch(Rectangle(
        (0.03, 0.7), 0.94, 0.2,
        facecolor=colors['background'],
        edgecolor='none',
        alpha=0.5,
        zorder=0
    ))
    
    # Add title with enhanced styling
    ax.text(
        0.5, 0.83, 
        'Intelligent Algorithmic Trading System Architecture',
        ha='center', va='center', 
        fontsize=24, 
        fontweight='bold', 
        color=colors['text_dark'],
        family='sans-serif'
    )
    
    # Add subtitle with paper reference
    ax.text(
        0.5, 0.77,
        'Based on Sezer & Ozbayoglu (2018) with attention mechanism enhancements',
        ha='center', va='center', 
        fontsize=14, 
        fontstyle='italic', 
        color='#64748b',
        family='sans-serif'
    )
    
    # Define component dimensions and positions with more space
    num_components = len(components)
    component_width = 0.12  # Increased width
    component_height = 0.45  # Slightly shorter to prevent overlap
    total_width = 0.88
    spacing = (total_width - (num_components * component_width)) / (num_components - 1)
    
    y_position = 0.45  # Vertical center of the plot
    
    # Draw components with a more modern, gradient look
    for i, component in enumerate(components):
        # Calculate x position
        x = 0.05 + i * (component_width + spacing)
        
        # Create shadow effect for depth
        ax.add_patch(Rectangle(
            (x+0.005, y_position-component_height/2-0.005), 
            component_width, 
            component_height,
            facecolor='#64748b', 
            alpha=0.15,
            zorder=1
        ))
        
        # Use component-specific color
        color = component['color']
        
        # Create main component rectangle
        ax.add_patch(Rectangle(
            (x, y_position-component_height/2), 
            component_width, 
            component_height,
            facecolor=color,
            edgecolor='white',
            linewidth=1.5,
            alpha=0.85,
            zorder=2
        ))
        
        # Add circular symbol at top
        circle_radius = 0.04
        ellipse = Ellipse(
            (x + component_width/2, y_position + component_height/3), 
            circle_radius*2, 
            circle_radius*2, 
            facecolor='white', 
            edgecolor='none', 
            zorder=3,
            alpha=0.95
        )
        ax.add_patch(ellipse)
        
        # Add symbol text with shadow effect for better visibility
        symbol_text = ax.text(
            x + component_width/2, 
            y_position + component_height/3,
            component['symbol'], 
            ha='center', 
            va='center', 
            fontsize=18, 
            fontweight='bold',
            color=color,
            zorder=5,
            family='monospace'
        )
        symbol_text.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='white')
        ])
        
        # Add component name (white bold text) with clear contrast
        name_text = ax.text(
            x + component_width/2, 
            y_position + 0.02,  # Adjust position
            component['name'], 
            ha='center', 
            va='center', 
            fontsize=12, 
            fontweight='bold', 
            color='white',
            wrap=True,
            zorder=4,
            family='sans-serif'
        )
        name_text.set_path_effects([
            PathEffects.withStroke(linewidth=1.5, foreground=(0, 0, 0, 0.7))
        ])
        
        # Add description with improved readability - now using newlines
        desc_text = ax.text(
            x + component_width/2, 
            y_position - component_height/3,
            component['desc'], 
            ha='center', 
            va='center', 
            fontsize=10, 
            fontweight='normal',
            color='white',
            linespacing=1.2,
            zorder=4,
            family='sans-serif'
        )
        desc_text.set_path_effects([
            PathEffects.withStroke(linewidth=1.5, foreground=(0, 0, 0, 0.7))
        ])
        
        # Add stylish connecting arrows with gradient
        if i < num_components - 1:
            arrow_x_start = x + component_width + 0.005
            arrow_x_end = x + component_width + spacing - 0.005
            
            # Create fancy arrow with improved styling
            arrow = FancyArrowPatch(
                (arrow_x_start, y_position), 
                (arrow_x_end, y_position),
                arrowstyle='-|>',
                mutation_scale=15,
                linewidth=2.5,
                color='white',
                zorder=5,
                connectionstyle="arc3,rad=0.0",
                alpha=0.8
            )
            ax.add_patch(arrow)
    
    # Add a data flow arrow with clear positioning
    dataflow_arrow = FancyArrowPatch(
        (0.2, 0.19),
        (0.8, 0.19),
        arrowstyle='-|>',
        mutation_scale=20,
        linewidth=3.5,
        color=colors['arrow'],
        zorder=5,
        connectionstyle="arc3,rad=0.0",
        alpha=0.7
    )
    ax.add_patch(dataflow_arrow)
    
    # Add "Data Flow" text with improved visibility
    # Background rectangle for data flow text
    text_bg = Rectangle(
        (0.47, 0.17),
        0.06,
        0.04,
        facecolor='white',
        alpha=0.7,
        zorder=5
    )
    ax.add_patch(text_bg)
    
    flow_text = ax.text(
        0.5, 
        0.19, 
        'Data Flow',
        ha='center', 
        va='center', 
        fontsize=14, 
        fontweight='bold', 
        color=colors['text_dark'],
        family='sans-serif',
        zorder=6
    )
    flow_text.set_path_effects([
        PathEffects.withStroke(linewidth=3, foreground='white')
    ])
    
    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout(pad=0)
    return fig