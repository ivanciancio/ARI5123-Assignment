"""
Intelligent Algorithmic Trading System

This Streamlit application implements an intelligent algorithmic trading system using
Convolutional Neural Networks (CNNs) based on the approach from Sezer & Ozbayoglu (2018).

The system allows users to:
1. Download historical stock data
2. Prepare and visualize data
3. Train and evaluate CNN models
4. Apply and compare trading strategies
5. Visualize performance results

Author: Ivan Ciancio
Date: March 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import datetime
import time
import torch
from sklearn.metrics import classification_report
from torchinfo import summary as torch_summary


# Import custom modules
from utils.data_loader import DataLoader
from utils.feature_engineering import FeatureEngineer
from utils.model import CNNTradingModel
from utils.trading_strategy import TradingStrategy
from utils.visualization import (
    plot_stock_data, plot_model_training_history, plot_model_evaluation,
    plot_portfolio_performance, plot_interactive_portfolio, plot_strategy_comparison,
    create_performance_report
)

# Set page configuration
st.set_page_config(
    page_title="Intelligent Algorithmic Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'strategy_results' not in st.session_state:
    st.session_state.strategy_results = None
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = None

# Define date format for British English
DATE_FORMAT = "%d/%m/%Y"

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border: 1px solid #E2E8F0;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# Define Streamlit components

def main():
    """Main function for the Streamlit app."""
    # Display header
    st.markdown('<div class="main-header">Intelligent Algorithmic Trading System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        This application implements an intelligent algorithmic trading system using Convolutional Neural Networks (CNNs).
        The approach is based on the paper "Financial Trading Model with Stock Bar Chart Image Time Series with Deep 
        Convolutional Neural Networks" by Sezer & Ozbayoglu (2018) with modern enhancements.
    </div>
    """, unsafe_allow_html=True)
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Introduction", "Data Preparation", "Model Training", "Trading Strategy", "Performance Analysis", "About"])
    
    # Display selected page
    if page == "Introduction":
        display_introduction()
    elif page == "Data Preparation":
        display_data_preparation()
    elif page == "Model Training":
        display_model_training()
    elif page == "Trading Strategy":
        display_trading_strategy()
    elif page == "Performance Analysis":
        display_performance_analysis()
    elif page == "About":
        display_about()

def display_introduction():
    """Display introduction page with project overview."""
    st.markdown('<div class="sub-header">Introduction</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("""
    ### Project Overview
    
    This project implements an intelligent algorithmic trading system using Convolutional Neural Networks (CNNs).
    The system is designed to analyse historical stock data, identify patterns, and make trading decisions.
    
    ### Key Features
    
    - **CNN-based Trading Model**: Utilises deep learning to identify patterns in stock price movements
    - **Technical Indicator Analysis**: Incorporates various technical indicators for enhanced feature extraction
    - **Automated Trading Strategy**: Implements a trading strategy based on model predictions
    - **Performance Evaluation**: Compares strategy performance against benchmarks
    
    ### Implementation Details
    
    The implementation is based on the paper "Financial Trading Model with Stock Bar Chart Image Time Series with Deep 
    Convolutional Neural Networks" by Sezer & Ozbayoglu (2018), with several modern enhancements:
    
    1. **Enhanced CNN Architecture**: Incorporates attention mechanisms and batch normalisation
    2. **Advanced Feature Engineering**: Uses a broader set of technical indicators and price transformations
    3. **Improved Trading Strategy**: Implements position sizing and risk management techniques
    4. **Comprehensive Evaluation**: Provides detailed performance metrics and visualisations
    
    ### How to Use This Application
    
    1. **Data Preparation**: Download and prepare historical stock data
    2. **Model Training**: Train and evaluate the CNN model
    3. **Trading Strategy**: Apply the trading strategy based on model predictions
    4. **Performance Analysis**: Analyse and compare strategy performance
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display improved architecture diagram
    st.markdown('<div class="sub-header">System Architecture</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    
    # Import the create_improved_architecture_diagram function
    from utils.visualization import create_improved_architecture_diagram
    
    # Create and display the diagram
    try:
        fig = create_improved_architecture_diagram()
        st.pyplot(fig)
    except Exception as e:
        # Fallback to the old diagram if there's an error
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            # Create a simple architecture diagram using matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Define the components
            components = [
                "Historical Stock Data", 
                "Data Preprocessing", 
                "Feature Engineering", 
                "CNN Model", 
                "Trading Signals", 
                "Trading Strategy", 
                "Performance Evaluation"
            ]
            
            # Create a horizontal flow diagram
            for i, component in enumerate(components):
                rect = plt.Rectangle((i, 0), 0.8, 0.8, facecolor='#3B82F6', alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
                ax.text(i + 0.4, 0.4, component, ha='center', va='center', color='white', fontweight='bold', wrap=True)
                
                # Add arrow to next component
                if i < len(components) - 1:
                    ax.arrow(i + 0.85, 0.4, 0.1, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
            
            # Hide axes
            ax.axis('off')
            ax.set_xlim(-0.2, len(components))
            ax.set_ylim(-0.2, 1)
            
            st.pyplot(fig)
            st.error(f"Could not display improved architecture diagram: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_data_preparation():
    """Display data preparation page with data download and visualization."""
    st.markdown('<div class="sub-header">Data Preparation</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    # Download data
    st.markdown("### Download Stock Data")
    
    st.markdown("""
    <div class="info-box">
        For accurate benchmarking against Sezer & Ozbayoglu (2018), we use the same date range from their paper:
        January 1, 2007 to December 31, 2017. This ensures a fair comparison with their results.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date range selection with benchmark paper's dates as defaults
        benchmark_end_date = datetime.datetime.strptime("31/12/2017", DATE_FORMAT).date()
        benchmark_start_date = datetime.datetime.strptime("01/01/2007", DATE_FORMAT).date()
        
        end_date = st.date_input(
            "End Date", 
            value=benchmark_end_date,
            max_value=datetime.datetime.now().date()
        )
        start_date = st.date_input(
            "Start Date", 
            value=benchmark_start_date,
            max_value=end_date
        )
    
    with col2:
        # Stock selection - Dow 30 components from the benchmark period
        ticker_options = [
            "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
            "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
            "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
        ]
        selected_ticker = st.selectbox(
            "Select a stock ticker (Dow 30 components from 2007-2017)",
            options=ticker_options
        )
        
        # Force download checkbox
        force_download = st.checkbox("Force fresh download")
    
    # Convert dates to string format
    start_date_str = start_date.strftime("%d/%m/%Y")
    end_date_str = end_date.strftime("%d/%m/%Y")
    
    # Download button
    if st.button("Download Data"):
        with st.spinner("Downloading data..."):
            # Initialize DataLoader
            data_loader = DataLoader()
            
            # Download data
            data = data_loader.download_dow30_data(start_date_str, end_date_str, force_download)
            
            # Store data in session state
            st.session_state.data = data
            
            # Display success message
            if selected_ticker in data:
                st.success(f"Successfully downloaded data for {selected_ticker}")
                # Extra info about benchmark alignment
                if (start_date_str == "01/01/2007" and end_date_str == "31/12/2017"):
                    st.info("✓ Using the exact date range from Sezer & Ozbayoglu (2018) paper for proper benchmarking.")
                else:
                    st.warning("⚠️ Not using the exact benchmark date range. Consider using 01/01/2007 to 31/12/2017 for a fair comparison.")
            else:
                st.error(f"Failed to download data for {selected_ticker}")
    
    # Benchmark info
    with st.expander("📚 Benchmark Information"):
        st.markdown("""
        ### Sezer & Ozbayoglu (2018) Paper Details
        
        **Date Range:**
        - Training Period: January 1, 2007 - December 31, 2015
        - Testing Period: January 1, 2016 - December 31, 2017
        
        **Model Configuration:**
        - 30-day sliding window for creating image-based input
        - Convolutional Neural Network architectures
        - Binary classification (buy/sell signals)
        
        **Performance Metrics Used:**
        - Classification accuracy (typically 53-58%)
        - Cumulative return (compared against buy & hold)
        - Sharpe ratio
        - Maximum drawdown
        
        Using the same data range and stock selection ensures a fair comparison between your implementation and the benchmark results.
        """)
        
    st.markdown('</div>', unsafe_allow_html=True)

def display_model_training():
    """Display model training page with model configuration and training."""
    st.markdown('<div class="sub-header">Model Training</div>', unsafe_allow_html=True)
    
    # Check if data is prepared
    if 'prepared_data' not in st.session_state:
        st.warning("Please prepare data first in the 'Data Preparation' section.")
        return
    
    # Get prepared data
    prepared_data = st.session_state.prepared_data
    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']
    X_val = prepared_data['X_val']
    y_val = prepared_data['y_val']
    X_test = prepared_data['X_test']
    y_test = prepared_data['y_test']
    ticker = prepared_data['ticker']
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio(
            "Model Architecture",
            options=["Simple CNN", "Advanced CNN with Attention"],
            index=1
        )
        
        epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=50, step=5)
    
    with col2:
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128],
            value=32
        )
        
        use_early_stopping = st.checkbox("Use Early Stopping", value=True)
    
    # Initialize model
    input_shape = X_train.shape[1:]
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training model... This may take a few minutes."):
            # Initialize model
            model = CNNTradingModel(input_shape)
            
            # Build model
            if model_type == "Simple CNN":
                model.build_simple_cnn()
                advanced = False
            else:
                model.build_advanced_cnn()
                advanced = True
            
            # Display model summary
            st.markdown("#### Model Summary")
            model_summary = []
            # Create a sample input tensor with the right shape
            sample_input = torch.zeros((1,) + model.input_shape)
            summary_str = str(torch_summary(model.model, input_size=sample_input.shape))
            model_summary.append(summary_str)
            st.code("\n".join(model_summary))
            
            # Train model
            history = model.train(
                X_train, y_train, X_val, y_val,
                advanced=advanced,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Store model and history in session state
            st.session_state.model = model
            st.session_state.history = history
            
            # Display training plots
            st.markdown("#### Training History")
            fig = plot_model_training_history(history)
            st.pyplot(fig)
            
            # Evaluate model
            st.markdown("#### Model Evaluation")
            
            # Make predictions
            y_pred_prob = model.predict(X_test)
            st.session_state.predictions = y_pred_prob
            
            # Display evaluation metrics
            metrics = model.evaluate(X_test, y_test)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.2%}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.2%}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
            
            # Display evaluation plots
            fig = plot_model_evaluation(y_test, y_pred_prob)
            st.pyplot(fig)
            
            # Save model
            model.save(f"{ticker}_model")
            st.success(f"Model trained and saved as {ticker}_model.h5")
    st.markdown('</div>', unsafe_allow_html=True)

def display_trading_strategy():
    """Display trading strategy page with strategy configuration and execution."""
    st.markdown('<div class="sub-header">Trading Strategy</div>', unsafe_allow_html=True)
    
    # Check if model is trained
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' section.")
        return
    
    # Check if predictions are available
    if 'predictions' not in st.session_state or st.session_state.predictions is None:
        st.warning("Model predictions are not available. Please train the model first.")
        return
    
    # Get prepared data and predictions
    prepared_data = st.session_state.prepared_data
    X_test = prepared_data['X_test']
    y_test = prepared_data['y_test']
    ticker = prepared_data['ticker']
    window_size = prepared_data['window_size']
    
    predictions = st.session_state.predictions
    
    # Get original stock data
    data = st.session_state.data
    df = data[ticker].copy()
    
    # Get test dates
    data_loader = DataLoader()
    test_dates = data_loader.get_test_dates(
        data, ticker, window_size=window_size,
        train_ratio=0.7, val_ratio=0.15
    )
    
    # Ensure test_dates matches the length of predictions
    if len(test_dates) > len(predictions):
        test_dates = test_dates[-len(predictions):]
    elif len(test_dates) < len(predictions):
        predictions = predictions[-len(test_dates):]
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Trading Strategy Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capital = st.number_input(
            "Initial Capital (£)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
        
    with col2:
        transaction_cost = st.slider(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            format="%.2f"
        ) / 100
    
    # Execute strategy button
    if st.button("Execute Trading Strategy"):
        with st.spinner("Executing trading strategy..."):
            # Initialize trading strategy
            strategy = TradingStrategy(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost
            )
            
            # Get test set prices
            test_prices = df.loc[test_dates]['Close'].values
            
            # Apply trading strategy
            strategy_results = strategy.apply_strategy(
                test_prices, predictions.flatten(), test_dates
            )
            
            # Apply buy and hold benchmark
            benchmark_results = strategy.apply_buy_and_hold(test_prices)
            
            # Apply random strategy benchmark
            random_results = strategy.apply_random_strategy(test_prices)
            
            # Store results in session state
            st.session_state.strategy_results = strategy_results
            st.session_state.benchmark_results = benchmark_results
            st.session_state.random_results = random_results
            st.session_state.strategy_portfolio = strategy.get_portfolio_values()
            st.session_state.benchmark_portfolio = strategy.get_portfolio_values()  # From buy and hold
            st.session_state.trades = strategy.get_trades()
            st.session_state.test_dates = test_dates
            
            # Display strategy performance
            st.markdown("#### Strategy Performance")
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Return",
                    f"{strategy_results['total_return']:.2%}",
                    f"{strategy_results['total_return'] - benchmark_results['total_return']:.2%}"
                )
            with col2:
                st.metric(
                    "Sharpe Ratio",
                    f"{strategy_results['sharpe_ratio']:.2f}",
                    f"{strategy_results['sharpe_ratio'] - benchmark_results['sharpe_ratio']:.2f}"
                )
            with col3:
                st.metric(
                    "Max Drawdown",
                    f"{strategy_results['max_drawdown']:.2%}",
                    f"{benchmark_results['max_drawdown'] - strategy_results['max_drawdown']:.2%}"
                )
            with col4:
                st.metric(
                    "Win Rate",
                    f"{strategy_results['win_rate']:.2%}"
                )
            
            # Display portfolio performance plot
            st.markdown("#### Portfolio Performance")
            portfolio_values = st.session_state.strategy_portfolio
            benchmark_values = st.session_state.benchmark_portfolio
            
            fig = plot_interactive_portfolio(
                portfolio_values, benchmark_values,
                test_dates, st.session_state.trades
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display strategy comparison
            st.markdown("#### Strategy Comparison")
            strategies_results = {
                "CNN Strategy": strategy_results,
                "Buy and Hold": benchmark_results,
                "Random Strategy": random_results
            }
            
            fig = plot_strategy_comparison(strategies_results)
            st.pyplot(fig)
            
            # Display trades
            st.markdown("#### Trading Activity")
            trades_df = st.session_state.trades
            
            if len(trades_df) > 0:
                st.dataframe(trades_df)
            else:
                st.info("No trades were executed during the test period.")
    st.markdown('</div>', unsafe_allow_html=True)

def display_performance_analysis():
    """Display performance analysis page with detailed performance metrics."""
    st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
    
    # Check if strategy results are available
    if 'strategy_results' not in st.session_state or st.session_state.strategy_results is None:
        st.warning("Please execute a trading strategy first in the 'Trading Strategy' section.")
        return
    
    # Get results
    strategy_results = st.session_state.strategy_results
    benchmark_results = st.session_state.benchmark_results
    random_results = st.session_state.random_results
    
    # Get ticker information from session state
    ticker = st.session_state.prepared_data['ticker'] if 'prepared_data' in st.session_state else "selected stock"
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Performance Report")
    
    
def display_performance_analysis():
    """Display performance analysis page with detailed performance metrics."""
    st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
    
    # Check if strategy results are available
    if 'strategy_results' not in st.session_state or st.session_state.strategy_results is None:
        st.warning("Please execute a trading strategy first in the 'Trading Strategy' section.")
        return
    
    # Get results
    strategy_results = st.session_state.strategy_results
    benchmark_results = st.session_state.benchmark_results
    random_results = st.session_state.random_results
    
    # Get ticker information from session state
    ticker = st.session_state.prepared_data['ticker'] if 'prepared_data' in st.session_state else "selected stock"
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Performance Report")
    
    # Create metrics display directly
    metrics = [
        ('Total Return', f"{strategy_results['total_return']:.2%}", 
         f"{benchmark_results['total_return']:.2%}" if benchmark_results else None),
        ('Annualized Return', f"{strategy_results['annualized_return']:.2%}", 
         f"{benchmark_results['annualized_return']:.2%}" if benchmark_results else None),
        ('Sharpe Ratio', f"{strategy_results['sharpe_ratio']:.2f}", 
         f"{benchmark_results['sharpe_ratio']:.2f}" if benchmark_results else None),
        ('Maximum Drawdown', f"{strategy_results['max_drawdown']:.2%}", 
         f"{benchmark_results['max_drawdown']:.2%}" if benchmark_results else None),
        ('Win Rate', f"{strategy_results['win_rate']:.2%}", 
         f"{benchmark_results['win_rate']:.2%}" if benchmark_results else None),
        ('Profit Factor', f"{strategy_results['profit_factor']:.2f}", 
         f"{benchmark_results['profit_factor']:.2f}" if benchmark_results else None),
        ('Number of Trades', f"{strategy_results['number_of_trades']}", 
         f"{benchmark_results['number_of_trades']}" if benchmark_results else None),
        ('Final Portfolio Value', f"£{strategy_results['final_value']:.2f}", 
         f"£{benchmark_results['final_value']:.2f}" if benchmark_results else None)
    ]
    
    # Create a DataFrame for better display
    if benchmark_results:
        metrics_df = pd.DataFrame(metrics, columns=["Metric", "Strategy", "Benchmark"])
    else:
        metrics_df = pd.DataFrame([(m[0], m[1]) for m in metrics], columns=["Metric", "Strategy"])
    
    # Display metrics in a styled table
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Add interpretation section
    st.markdown("""
    ### Interpretation
    
    - The **Sharpe ratio** measures the risk-adjusted return, with higher values indicating better risk-adjusted performance.
      A Sharpe ratio above 1.0 is generally considered good, while above 2.0 is excellent.
      
    - The **maximum drawdown** represents the largest peak-to-trough decline and is a measure of downside risk.
      Lower values are better.
      
    - The **win rate** shows the percentage of profitable trades, while the **profit factor** is the ratio of gross profits to gross losses.
      A profit factor above 1.0 indicates a profitable strategy.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)    
        
    # Results discussion
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Results Discussion")
    
    # Calculate outperformance
    outperformance = strategy_results['total_return'] - benchmark_results['total_return']
    risk_diff = strategy_results['max_drawdown'] - benchmark_results['max_drawdown']
    sharpe_diff = strategy_results['sharpe_ratio'] - benchmark_results['sharpe_ratio']
    
    # Generate discussion based on results
    st.markdown("#### Strategy Analysis")
    
    if outperformance > 0:
        st.markdown(f"""
        The CNN-based trading strategy has outperformed the buy-and-hold benchmark by 
        **{outperformance:.2%}** in terms of total return, demonstrating the effectiveness of the model 
        in identifying profitable trading opportunities in the {ticker} stock.
        """)
    else:
        st.markdown(f"""
        The CNN-based trading strategy has underperformed the buy-and-hold benchmark by 
        **{-outperformance:.2%}** in terms of total return. This suggests that either the model was 
        not able to capture meaningful patterns in the {ticker} stock, or that the 
        selected time period was particularly challenging for active trading strategies.
        """)
    
    # Risk analysis
    st.markdown("#### Risk Analysis")
    
    if risk_diff < 0:
        st.markdown(f"""
        In terms of risk, the strategy demonstrated better drawdown characteristics, with a maximum 
        drawdown of **{strategy_results['max_drawdown']:.2%}** compared to the benchmark's 
        **{benchmark_results['max_drawdown']:.2%}**. This indicates that the strategy was effective 
        in managing downside risk, which is particularly valuable in volatile markets.
        """)
    else:
        st.markdown(f"""
        The strategy showed a higher maximum drawdown of **{strategy_results['max_drawdown']:.2%}** 
        compared to the benchmark's **{benchmark_results['max_drawdown']:.2%}**. This suggests that 
        while attempting to capture additional returns, the strategy took on additional risk, 
        which should be considered when evaluating its overall performance.
        """)
    
    # Trading efficiency
    st.markdown("#### Trading Efficiency")
    
    # Use transaction_cost from session state if available, or default to 0.001
    transaction_cost = 0.001
    if 'transaction_cost' in st.session_state:
        transaction_cost = st.session_state.transaction_cost
    
    st.markdown(f"""
    The strategy executed a total of **{strategy_results['number_of_trades']}** trades, with a win rate 
    of **{strategy_results['win_rate']:.2%}** and a profit factor of **{strategy_results['profit_factor']:.2f}**. 
    The transaction costs had a significant impact on the overall performance, reducing the gross returns 
    by approximately **{strategy_results['number_of_trades'] * 2 * transaction_cost:.2%}**.
    """)
    
    # Comparison to random strategy
    st.markdown("#### Comparison to Random Strategy")
    
    random_diff = strategy_results['total_return'] - random_results['total_return']
    
    if random_diff > 0:
        st.markdown(f"""
        Importantly, the CNN-based strategy outperformed the random strategy by **{random_diff:.2%}**, 
        suggesting that the model is capturing meaningful patterns in the price data rather than 
        simply benefiting from general market movements or randomness.
        """)
    else:
        st.markdown(f"""
        Concerningly, the CNN-based strategy underperformed the random strategy by **{-random_diff:.2%}**. 
        This raises questions about whether the model is capturing meaningful patterns or if the 
        performance could be attributed to chance. Further refinement and testing may be necessary.
        """)
    
    # Conclusion and recommendations
    st.markdown("#### Conclusion and Recommendations")
    
    if strategy_results['sharpe_ratio'] > 1.0 and outperformance > 0:
        st.markdown("""
        Overall, the CNN-based trading strategy demonstrates promising results, with a positive risk-adjusted 
        return as measured by the Sharpe ratio and outperformance relative to passive benchmarks. 
        
        **Recommendations for further improvement:**
        
        1. **Feature Engineering**: Explore additional technical indicators or alternative data sources
        2. **Model Optimization**: Fine-tune model hyperparameters and architecture
        3. **Risk Management**: Implement more sophisticated position sizing and stop-loss mechanisms
        4. **Market Regime Awareness**: Develop filters to avoid trading during unfavorable market conditions
        5. **Ensemble Approaches**: Combine predictions with other models for more robust signals
        """)
    else:
        st.markdown("""
        The CNN-based trading strategy shows mixed results, with challenges in consistently outperforming 
        passive benchmarks on a risk-adjusted basis.
        
        **Recommendations for improvement:**
        
        1. **Model Reevaluation**: Review the model architecture and training process
        2. **Feature Selection**: Identify more predictive features and remove noise
        3. **Trading Rules Refinement**: Adjust entry and exit criteria to improve win rate
        4. **Transaction Cost Optimization**: Reduce trading frequency to minimize costs
        5. **Alternative Approaches**: Consider other AI/ML techniques that may be more suitable
        """)
    st.markdown('</div>', unsafe_allow_html=True)

def display_about():
    """Display about page with project information."""
    st.markdown('<div class="sub-header">About This Project</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("""
    ### Project Information
    
    This project was developed as part of the Intelligent Algorithmic Trading Assignment (ARI5123, Assignment 2)
    for the University of Malta's Department of Artificial Intelligence.
    
    ### Methodology
    
    The methodology is based on the following paper:
    
    > Sezer, O. B., & Ozbayoglu, A. M. (2018). Financial Trading Model with Stock Bar Chart Image Time Series with Deep Convolutional Neural Networks. arXiv preprint arXiv:1809.01560.
    
    The original approach was enhanced with:
    
    1. Advanced CNN architecture with attention mechanisms
    2. Extended feature engineering
    3. Improved trading strategy implementation
    4. Comprehensive performance evaluation
    
    ### Technologies Used
    
    - **Python**: Core programming language
    - **PyTorch**: Deep learning framework
    - **Pandas/NumPy**: Data manipulation
    - **Matplotlib/Plotly**: Data visualization
    - **Streamlit**: Web application framework
    - **yfinance**: Financial data API
    
    ### Limitations and Future Work
    
    **Limitations:**
    
    - Limited to historical backtesting, not live trading
    - Uses only technical indicators, not fundamental data
    - Transaction costs are simplified
    - Market impact and liquidity concerns are not modeled
    
    **Future Work:**
    
    - Incorporate alternative data sources
    - Implement ensemble methods for more robust predictions
    - Develop adaptive parameter optimization
    - Integrate with live trading APIs
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### References")
    
    st.markdown("""
    1. Sezer, O. B., & Ozbayoglu, A. M. (2018). Financial trading model with stock bar chart image time series with deep convolutional neural networks. arXiv preprint arXiv:1809.01560.
    
    2. Bao, W., Yue, J., & Rao, Y. (2017). A deep learning framework for financial time series using stacked autoencoders and long-short term memory. PloS one, 12(7), e0180944.
    
    3. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654-669.
    
    4. Jiang, Z., Xu, D., & Liang, J. (2017). A deep reinforcement learning framework for the financial portfolio management problem. arXiv preprint arXiv:1706.10059.
    
    5. Di Persio, L., & Honchar, O. (2016). Artificial neural networks architectures for stock price prediction: Comparisons and applications. International Journal of Circuits, Systems and Signal Processing, 10, 403-413.
    
    6. Tsantekidis, A., Passalis, N., Tefas, A., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2017, July). Forecasting stock prices from the limit order book using convolutional neural networks. In 2017 IEEE 19th Conference on Business Informatics (CBI) (Vol. 1, pp. 7-12). IEEE.
    
    7. Hu, Z., Liu, W., Bian, J., Liu, X., & Liu, T. Y. (2018). Listening to chaotic whispers: A deep learning framework for news-oriented stock trend prediction. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining (pp. 261-269).
    
    8. Ding, X., Zhang, Y., Liu, T., & Duan, J. (2015). Deep learning for event-driven stock prediction. In Twenty-fourth international joint conference on artificial intelligence.
    
    9. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
    
    10. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()