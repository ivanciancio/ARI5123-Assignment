"""
Visualisation Module

This module provides functions for visualising trading data, model performance, and strategy results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse
import matplotlib.patheffects as PathEffects

# Set style for plots
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.style.use('seaborn-v0_8-darkgrid')

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
                            # If date not found, skip
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
                            # If date not found, skip
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
        yaxis_title="Portfolio Value ($)",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        template="plotly_white"
    )
    
    # Format hover tooltip
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: $%{y:.2f}<extra></extra>"
    )
    
    return fig

def create_improved_architecture_diagram():
    """
    Create a visually enhanced architecture diagram for the trading system
    with improved text clarity and proper spacing.
    
    Returns:
        Matplotlib figure
    """
    # Create figure with a premium white background
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    ax.set_facecolor('white')
    
    # Remove axes
    ax.axis('off')
    
    # Define an improved color palette
    colors = {
        'primary': '#3b82f6',
        'secondary': '#1e40af',
        'accent1': '#34d399',
        'accent2': '#f97316',
        'background': '#f8fafc',
        'text_dark': '#0f172a',
        'text_light': '#f8fafc',
        'gradient_top': '#3b82f6',
        'gradient_bottom': '#1e40af',
        'arrow': '#0369a1',
        'shadow': (0, 0, 0, 0.5)
    }
    
    # Component details with descriptions
    components = [
        {
            'name': 'Historical\nStock Data',
            'desc': 'OHLCV data from\nEODHD',
            'symbol': 'D',
            'color': '#4f46e5'
        },
        {
            'name': 'Data\nPreprocessing',
            'desc': 'Normalisation &\nsequence creation',
            'symbol': 'P',
            'color': '#8b5cf6'
        },
        {
            'name': 'Feature\nEngineering',
            'desc': 'Bar chart images\n30x30 pixels',
            'symbol': 'F',
            'color': '#3b82f6'
        },
        {
            'name': 'CNN Model',
            'desc': 'Deep learning\nwith attention',
            'symbol': 'M',
            'color': '#0ea5e9'
        },
        {
            'name': 'Trading\nSignals',
            'desc': 'Buy/sell probability\npredictions',
            'symbol': 'S',
            'color': '#14b8a6'
        },
        {
            'name': 'Trading\nStrategy',
            'desc': 'Position sizing &\nrisk management',
            'symbol': 'T',
            'color': '#10b981'
        },
        {
            'name': 'Performance\nEvaluation',
            'desc': 'Return & risk\nmetrics analysis',
            'symbol': 'E',
            'color': '#22c55e'
        }
    ]
    
    # Add a background panel
    ax.add_patch(Rectangle(
        (0.03, 0.15), 0.94, 0.65, 
        facecolor=colors['background'],
        edgecolor='#e2e8f0',
        linewidth=1,
        alpha=0.8,
        zorder=0
    ))
    
    # Add a header band
    ax.add_patch(Rectangle(
        (0.03, 0.7), 0.94, 0.2,
        facecolor=colors['background'],
        edgecolor='none',
        alpha=0.5,
        zorder=0
    ))
    
    # Add title
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
        'Based on Sezer & Ozbayoglu (2018) with enhanced regularisation and signal processing',
        ha='center', va='center', 
        fontsize=14, 
        fontstyle='italic', 
        color='#64748b',
        family='sans-serif'
    )
    
    # Define component dimensions and positions with more space
    num_components = len(components)
    component_width = 0.12
    component_height = 0.45
    total_width = 0.88
    spacing = (total_width - (num_components * component_width)) / (num_components - 1)
    
    y_position = 0.45  # Vertical center of the plot
    
    # Draw components
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
        
        # Add symbol text
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
        
        # Add component name (white bold text)
        name_text = ax.text(
            x + component_width/2, 
            y_position + 0.02,
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
        
        # Add description
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
        
        # Add connecting arrows with gradient
        if i < num_components - 1:
            arrow_x_start = x + component_width + 0.005
            arrow_x_end = x + component_width + spacing - 0.005
            
            # Create arrow
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
    
    # Add a data flow arrow
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
    
    # Add "Data Flow" text
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