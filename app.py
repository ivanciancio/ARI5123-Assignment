"""
Intelligent Algorithmic Trading System

This Streamlit application implements an intelligent algorithmic trading system using
Convolutional Neural Networks (CNNs) based on the approach from Sezer & Ozbayoglu (2018).

The system allows users to:
1. Download historical stock data
2. Prepare and visualise data
3. Train and evaluate CNN models
4. Apply and compare trading strategies
5. Visualise performance results

ARI5123 - Ivan Ciancio
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchinfo import summary as torch_summary
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import os
from datetime import datetime

# Set page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Intelligent Algorithmic Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import custom modules
from utils.model import SimpleCNN, AdvancedCNN
from utils.data_loader import DataLoader
from utils.feature_engineering import FeatureEngineer
from utils.model import CNNTradingModel
from utils.trading_strategy import TradingStrategy
from utils.visualization import (
    plot_model_training_history, plot_interactive_portfolio,
    create_improved_architecture_diagram
)

# Initialise session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'prepared_data' not in st.session_state:
    st.session_state.prepared_data = None
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
    
    /* Updated section styling for dark mode compatibility */
    .section {
        background-color: var(--background-color, #F8FAFC);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color, #E2E8F0);
        color: var(--text-color, #1a202c);
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .section {
            background-color: #2D3748 !important;
            border-color: #4A5568 !important;
            color: #E2E8F0 !important;
        }
        .main-header {
            color: #63B3ED !important;
        }
        .sub-header {
            color: #90CDF4 !important;
        }
        .info-box {
            background-color: #1A365D !important;
            border-left-color: #63B3ED !important;
            color: #E2E8F0 !important;
        }
        .metric-card {
            background-color: #2D3748 !important;
            color: #E2E8F0 !important;
        }
        .metric-value {
            color: #90CDF4 !important;
        }
    }
    
    /* Force text visibility in all modes */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: inherit !important;
    }
    
    .info-box {
        background-color: var(--info-bg, #EFF6FF);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--info-border, #3B82F6);
        margin-bottom: 1rem;
        color: var(--info-text, #1a202c);
    }
    .metric-card {
        background-color: var(--card-bg, #FFFFFF);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        text-align: center;
        color: var(--card-text, #1a202c);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--metric-color, #2563EB);
    }
    .metric-label {
        font-size: 0.9rem;
        color: var(--label-color, #64748B);
    }
    
    /* Streamlit-specific dark mode overrides */
    [data-theme="dark"] .section {
        background-color: #2D3748 !important;
        border-color: #4A5568 !important;
        color: #E2E8F0 !important;
    }
    
    [data-theme="dark"] .info-box {
        background-color: #1A365D !important;
        border-left-color: #63B3ED !important;
        color: #E2E8F0 !important;
    }
    
    [data-theme="dark"] .metric-card {
        background-color: #2D3748 !important;
        border: 1px solid #4A5568 !important;
        color: #E2E8F0 !important;
    }
    
    [data-theme="dark"] .metric-value {
        color: #90CDF4 !important;
    }
    
    [data-theme="dark"] .metric-label {
        color: #A0AEC0 !important;
    }
    
    /* Additional Streamlit component overrides for dark mode */
    [data-theme="dark"] .stMarkdown h1,
    [data-theme="dark"] .stMarkdown h2,
    [data-theme="dark"] .stMarkdown h3,
    [data-theme="dark"] .stMarkdown h4,
    [data-theme="dark"] .stMarkdown p,
    [data-theme="dark"] .stMarkdown div {
        color: #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for the Streamlit app."""
    st.markdown('<div class="main-header">Intelligent Algorithmic Trading System - ARI5123</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        An intelligent algorithmic trading system has been implemented using Convolutional Neural Networks (CNNs). The approach is based upon Sezer & Ozbayoglu (2018), with modern enhancements having been incorporated.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Experimental Workflow:**
    1. **Data Preparation → CNN Training → Trading Strategy → Performance Analysis**
    2. **Multiple Configurations:** Simple/Advanced CNN architectures with Fixed/Dynamic signal processing
    3. **Benchmark Comparison:** Evaluation against Sezer & Ozbayoglu (2018) methodology
    4. **Important Note:** Results vary between runs due to neural network stochasticity
    """)
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation Menu")
    
    page = st.sidebar.radio("Select Analysis Type", [
        "🏠 Introduction", 
        "📊 Data Preparation (Single Stock)",
        "🤖 Model Training (Single Stock)", 
        "📈 Trading Strategy (Single Stock)",
        "📋 Performance Analysis (Single Stock)",
        "🚀 Comprehensive Benchmark Experiment"
    ])
    
    # Display selected page
    if page == "🏠 Introduction":
        display_introduction()
    elif page == "📊 Data Preparation (Single Stock)":
        display_data_preparation()
    elif page == "🤖 Model Training (Single Stock)":
        display_model_training()
    elif page == "📈 Trading Strategy (Single Stock)":
        display_trading_strategy()
    elif page == "📋 Performance Analysis (Single Stock)":
        display_performance_analysis()
    elif page == "🚀 Comprehensive Benchmark Experiment":
        display_batch_experiment_page()

def display_introduction():
    """Display introduction page with project overview."""
    st.markdown('<div class="sub-header">Introduction</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("""
    ### 📊 Project Overview

    This project implements an intelligent algorithmic trading system using Convolutional Neural Networks (CNNs).
    The system is designed to analyse historical stock data, identify patterns, and make trading decisions.

    ### ⭐ Key Features

    - **CNN-based Trading Model**: Utilises deep learning to identify patterns in stock price movements
    - **Technical Indicator Analysis**: Incorporates various technical indicators for enhanced feature extraction
    - **Automated Trading Strategy**: Implements a trading strategy based on model predictions
    - **Performance Evaluation**: Compares strategy performance against benchmarks

    ### 🧪 Experimental Framework

    **CNN-BI Implementation Based on Sezer & Ozbayoglu (2018):**
    - **Benchmark Paper Target:** 7.20% annualised return (2007-2012 period)
    - **Multiple Architectures:** Simple and Advanced CNN configurations available
    - **Signal Processing:** Fixed and Dynamic threshold methods implemented  
    - **Test Periods:** 2007-2012 (crisis) and 2012-2017 (bull market) evaluation

    **Important:** Neural network training introduces variability - results will differ between runs. 
    Multiple experiments recommended for robust performance assessment.

    ### 🔧 Implementation Details

    The implementation is based on the paper "Financial Trading Model with Stock Bar Chart Image Time Series with Deep 
    Convolutional Neural Networks" by Sezer & Ozbayoglu (2018), with several modern enhancements:

    1. **Enhanced CNN Architecture**: Incorporates attention mechanisms and batch normalisation
    2. **Advanced Feature Engineering**: Uses a broader set of technical indicators and price transformations
    3. **Improved Trading Strategy**: Implements position sizing and risk management techniques
    4. **Comprehensive Evaluation**: Provides detailed performance metrics and visualisations

    ### 📋 How to Use This Application

    1. **Data Preparation**: Download and prepare historical stock data
    2. **Model Training**: Train and evaluate the CNN model
    3. **Trading Strategy**: Apply the trading strategy based on model predictions
    4. **Performance Analysis**: Analyse and compare strategy performance
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display architecture diagram
    st.markdown('<div class="sub-header">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    
    try:
        fig = create_improved_architecture_diagram()
        st.pyplot(fig)
    except Exception as e:
        # Fallback simple diagram
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            components = [
                "Historical Stock Data", 
                "Data Preprocessing", 
                "Feature Engineering", 
                "CNN Model", 
                "Trading Signals", 
                "Trading Strategy", 
                "Performance Evaluation"
            ]
            
            for i, component in enumerate(components):
                rect = plt.Rectangle((i, 0), 0.8, 0.8, facecolor='#3B82F6', alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
                ax.text(i + 0.4, 0.4, component, ha='center', va='center', color='white', fontweight='bold', wrap=True)
                
                if i < len(components) - 1:
                    ax.arrow(i + 0.85, 0.4, 0.1, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
            
            ax.axis('off')
            ax.set_xlim(-0.2, len(components))
            ax.set_ylim(-0.2, 1)
            
            st.pyplot(fig)
            st.error(f"Could not display architecture diagram: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_data_preparation():
    """Display data preparation page with benchmark data download and visualisation."""
    st.markdown('<div class="sub-header">Data Preparation</div>', unsafe_allow_html=True)
    
    # Check if EODHD is configured
    eodhd_configured = False
    try:
        from utils.eodhdapi import get_client
        client = get_client()
        eodhd_configured = True
        st.success("✅ EODHD API configured successfully")
    except Exception as e:
        st.error(f"❌ EODHD API Error: {str(e)}")
        st.info("""
        To use this application, you need an EODHD API key:
        1. Sign up at https://eodhd.com
        2. Get API key from the dashboard
        3. Add it to Streamlit secrets as EODHD_API_KEY
        """)
        st.stop()
        return
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Download Benchmark Data")
    
    st.markdown("""
    <div class="info-box">
        For accurate benchmarking against Sezer & Ozbayoglu (2018), the same data range and methodology are used:
        - Full dataset: January 1997 to December 2017
        - Test Period 1: 2007-2012 (includes financial crisis)
        - Test Period 2: 2012-2017 (bull market period)
        - Data Source: EODHD Historical Data API
    </div>
    """, unsafe_allow_html=True)
    
    # Benchmark period selection
    test_period = st.selectbox(
        "Select Benchmark Test Period",
        options=["2007-2012", "2012-2017"],
        help="Choose the test period for benchmarking against the paper"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_ticker = st.selectbox(
            "Select Stock (Dow 30 from benchmark period)",
            options=["AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
                    "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
                    "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"]
        )
        
        force_download = st.checkbox("Force fresh download")
    
    with col2:
        st.markdown("#### Benchmark Information")
        if test_period == "2007-2012":
            st.info("""
            **Period 1: 2007-2012**
            - Training: 1997-2006
            - Testing: 2007-2012
            - Includes 2008 financial crisis
            - Volatile/bear market conditions
            """)
        else:
            st.info("""
            **Period 2: 2012-2017** 
            - Training: 1997-2011
            - Testing: 2012-2017
            - Bull market period
            - Lower volatility
            """)
    
    # Download and prepare buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("1. Download Benchmark Data"):
            # Clear all downstream results when downloading new data
            for key in ['model', 'history', 'predictions', 'strategy_results', 'benchmark_results', 
                       'random_results', 'strategy_portfolio', 'benchmark_portfolio', 'trades', 'test_dates']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Create a container for detailed progress
            progress_container = st.container()
            
            with progress_container:
                st.markdown("### 📊 Download Progress (Using EODHD API)")
                overall_progress = st.progress(0)
                status_text = st.empty()
                
                # Create columns for stock status display
                st.markdown("#### Stock Download Status")
                col1_stocks, col2_stocks, col3_stocks, col4_stocks, col5_stocks = st.columns(5)
                status_placeholders = {}
                
                dow30_tickers = [
                    "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
                    "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
                    "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
                ]
                
                # Create placeholder for each ticker
                for i, ticker in enumerate(dow30_tickers):
                    col = [col1_stocks, col2_stocks, col3_stocks, col4_stocks, col5_stocks][i % 5]
                    with col:
                        status_placeholders[ticker] = st.empty()
                        status_placeholders[ticker].markdown(f"⏳ **{ticker}**")
                
                # Statistics placeholders
                st.markdown("---")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    success_counter = st.empty()
                    success_counter.metric("✅ Successful", "0")
                with stats_col2:
                    failed_counter = st.empty()
                    failed_counter.metric("❌ Failed", "0")
                with stats_col3:
                    api_info = st.empty()
                    api_info.info("🌐 Using EODHD API")
                
                # Progress callback that updates individual stock status
                successful_count = 0
                failed_count = 0
                
                def detailed_progress_callback(current, total, ticker, status, extra_info=None):
                    nonlocal successful_count, failed_count
                    
                    # Update overall progress
                    progress = current / total
                    overall_progress.progress(progress)
                    
                    # Update status text
                    if "Waiting" in status:
                        status_text.text(f"⏸️ {status}")
                    else:
                        status_text.text(f"Processing {ticker} ({current}/{total})... {status}")
                    
                    # Update individual ticker status
                    if ticker in status_placeholders:
                        if "Success" in status:
                            status_placeholders[ticker].markdown(f"✅ **{ticker}**")
                            successful_count = current - failed_count
                            success_counter.metric("✅ Successful", str(successful_count))
                        elif "Failed" in status:
                            status_placeholders[ticker].markdown(f"❌ **{ticker}**")
                            failed_count += 1
                            failed_counter.metric("❌ Failed", str(failed_count))
                        elif "downloading" in status:
                            status_placeholders[ticker].markdown(f"⏬ **{ticker}**")
                        elif "Retrying" in status:
                            status_placeholders[ticker].markdown(f"🔄 **{ticker}**")
                        elif "Cached" in status:
                            status_placeholders[ticker].markdown(f"📦 **{ticker}**")
                            successful_count += 1
                            success_counter.metric("✅ Successful", str(successful_count))
                    
                    # Update API info if provided
                    if extra_info and "delay" in extra_info:
                        api_info.info(f"🌐 EODHD API - Next request in: {extra_info['delay']:.1f}s")
                
                # Initialise data loader and download
                data_loader = DataLoader()
                
                try:
                    with st.spinner("Initialising EODHD download..."):
                        # Use higher batch size for EODHD
                        data = data_loader.download_benchmark_data(
                            force_download=force_download,
                            batch_size=10,  
                            progress_callback=detailed_progress_callback
                        )
                        
                        st.session_state.data = data
                        
                        # Final summary
                        if data:
                            success_rate = (successful_count / len(dow30_tickers)) * 100
                            st.success(f"""
                            ✅ **Download Complete!**
                            - Successfully downloaded: {successful_count}/{len(dow30_tickers)} stocks ({success_rate:.1f}%)
                            - Failed downloads: {failed_count}
                            - Data source: EODHD Historical Data API
                            - Cache location: data/benchmark_dow30_eodhd_*.pkl
                            """)
                            
                            if selected_ticker in data:
                                st.info(f"""
                                **{selected_ticker} Data Details:**
                                - Start Date: {data[selected_ticker].index[0].strftime('%Y-%m-%d')}
                                - End Date: {data[selected_ticker].index[-1].strftime('%Y-%m-%d')}
                                - Total Trading Days: {len(data[selected_ticker])}
                                """)
                        else:
                            st.error("❌ No data was successfully downloaded. Please check EODHD API key and quota.")
                            
                except Exception as e:
                    st.error(f"❌ Error during download: {str(e)}")
                    if "API" in str(e) or "key" in str(e).lower():
                        st.info("""
                        💡 **Troubleshooting Tips:**
                        1. Check EODHD API key is valid
                        2. Verify API quota hasn't been exceeded
                        3. Try downloading fewer stocks if on a free plan
                        4. Check internet connection
                        5. Verify the API key is in .streamlit/secrets.toml
                        """)
                    st.exception(e)
    
    with col2:
        prepare_disabled = True
        if 'data' in st.session_state and st.session_state.data is not None:
            if selected_ticker in st.session_state.data:
                prepare_disabled = False
        
        if st.button("2. Prepare Benchmark Features", disabled=prepare_disabled):
            if st.session_state.data is not None and selected_ticker in st.session_state.data:
                # Clear downstream results when preparing new features
                for key in ['model', 'history', 'predictions', 'strategy_results', 'benchmark_results', 
                           'random_results', 'strategy_portfolio', 'benchmark_portfolio', 'trades', 'test_dates']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                with st.spinner("Preparing benchmark features..."):
                    try:
                        data_loader = DataLoader()
                        feature_engineer = FeatureEngineer()
                        
                        splits = data_loader.get_benchmark_splits(
                            st.session_state.data, 
                            selected_ticker, 
                            test_period
                        )
                        
                        prepared_data = feature_engineer.prepare_benchmark_features(
                            splits['train_data'],
                            splits['test_data'],
                            selected_ticker
                        )
                        
                        prepared_data['test_period'] = test_period
                        prepared_data['train_period'] = splits['train_dates']
                        prepared_data['test_period_dates'] = splits['test_dates']
                        
                        st.session_state.prepared_data = prepared_data
                        
                        # Handle validation data display
                        if prepared_data['X_val'] is not None:
                            val_samples_text = f", {len(prepared_data['X_val'])} validation"
                            methodology_text = "Ready for CNN training with validation split!"
                        else:
                            val_samples_text = " (no validation split - following exact paper methodology)"
                            methodology_text = "Ready for CNN training using exact benchmark methodology!"

                        st.success(f"""
                        ✅ Benchmark features prepared for {selected_ticker} ({test_period})!

                        **Training Data:**
                        - Period: {splits['train_dates'][0]} to {splits['train_dates'][1]}
                        - Samples: {len(prepared_data['X_train'])} training{val_samples_text}

                        **Testing Data:**
                        - Period: {splits['test_dates'][0]} to {splits['test_dates'][1]}
                        - Samples: {len(prepared_data['X_test'])} testing samples

                        **Label Distribution:**
                        - Hold (0): {np.sum(prepared_data['y_train'] == 0)} samples
                        - Buy (1): {np.sum(prepared_data['y_train'] == 1)} samples
                        - Sell (2): {np.sum(prepared_data['y_train'] == 2)} samples

                        {methodology_text}
                        """)
                        
                    except Exception as e:
                        st.error(f"Error preparing benchmark features: {str(e)}")
                        st.exception(e)
            else:
                st.error("Please download data first.")
    
    # Show data visualisation if available
    if 'data' in st.session_state and st.session_state.data is not None and selected_ticker in st.session_state.data:
        st.markdown("### Data Visualisation")
        
        ticker_data = st.session_state.data[selected_ticker]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(ticker_data.index, ticker_data['Close'], label='Close Price', color='blue', linewidth=1)
        
        # Highlight benchmark periods
        if test_period == "2007-2012":
            ax1.axvspan(pd.Timestamp('2007-01-01'), pd.Timestamp('2012-12-31'), 
                       alpha=0.3, color='red', label='Test Period (2007-2012)')
            ax1.axvspan(pd.Timestamp('1997-01-01'), pd.Timestamp('2006-12-31'), 
                       alpha=0.3, color='green', label='Training Period (1997-2006)')
        else:
            ax1.axvspan(pd.Timestamp('2012-01-01'), pd.Timestamp('2017-12-31'), 
                       alpha=0.3, color='red', label='Test Period (2012-2017)')
            ax1.axvspan(pd.Timestamp('1997-01-01'), pd.Timestamp('2011-12-31'), 
                       alpha=0.3, color='green', label='Training Period (1997-2011)')
        
        ax1.set_title(f"{selected_ticker} Stock Price - Benchmark Periods (Source: EODHD)")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(ticker_data.index, ticker_data['Volume'], alpha=0.6, color='purple', width=1)
        ax2.set_title(f"{selected_ticker} Trading Volume")
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Show preparation status
    if 'prepared_data' in st.session_state and st.session_state.prepared_data is not None:
        st.markdown("### ✅ Benchmark Data Preparation Status")
        prepared = st.session_state.prepared_data
        
        # Handle validation samples display
        if prepared['X_val'] is not None:
            val_samples_text = f"- Validation Samples: {len(prepared['X_val'])}\n    "
        else:
            val_samples_text = "- Validation: None (following exact paper methodology)\n    "
        
        st.success(f"""
        **Benchmark preparation complete for {prepared['ticker']}!**
        
        - Test Period: {prepared.get('test_period', 'Unknown')}
        - Training Samples: {len(prepared['X_train'])}
        {val_samples_text}- Testing Samples: {len(prepared['X_test'])}
        - Image Size: {prepared['X_train'].shape[1]}x{prepared['X_train'].shape[2]} pixels
        - Prediction Horizon: {prepared.get('prediction_horizon', 5)} days
        - Data Source: EODHD Historical Data API
        
        Ready for model training and benchmark comparison!
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_model_training():
    """Display model training page with model and settings."""
    st.markdown('<div class="sub-header">Model Training</div>', unsafe_allow_html=True)
    
    if 'prepared_data' not in st.session_state or st.session_state.prepared_data is None:
        st.warning("Please prepare data first in the 'Data Preparation' section.")
        return
    
    prepared_data = st.session_state.prepared_data
    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']
    X_val = prepared_data['X_val']
    y_val = prepared_data['y_val']
    X_test = prepared_data['X_test']
    y_test = prepared_data['y_test']
    ticker = prepared_data['ticker']
    
    # Ensure data is in the right shape for CNN
    if len(X_train.shape) == 3 and X_train.shape[1:] == (30, 30):
        X_train = X_train.reshape(X_train.shape[0], 1, 30, 30)
        
        # Handle validation data - it might be None
        if X_val is not None:
            X_val = X_val.reshape(X_val.shape[0], 1, 30, 30)
        
        X_test = X_test.reshape(X_test.shape[0], 1, 30, 30)
        
        # Update prepared_data
        prepared_data['X_train'] = X_train
        prepared_data['X_val'] = X_val  # This might be None, which is fine
        prepared_data['X_test'] = X_test
        st.session_state.prepared_data = prepared_data
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Model Configuration")
    
    # Handle validation data display
    if X_val is not None:
        val_info = f"- Validation samples: {X_val.shape[0]}\n"
    else:
        val_info = "- Validation: None (following exact paper methodology)\n"

    st.info(f"""
    **Data Summary for {ticker}**
    - Training samples: {X_train.shape[0]}
    {val_info}- Test samples: {X_test.shape[0]}
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio(
            "Model Architecture",
            options=["Simple CNN", "Advanced CNN"],
            index=0,  # Simple CNN default
            help="Simple CNN: Optimised for stability with enhanced regularisation - recommended for most use cases. Advanced CNN: More complex architecture with batch normalisation but greater variability."
        )
        
        epochs = st.slider("Training Epochs", min_value=10, max_value=200, value=100, step=5)
        
        st.markdown("#### Regularisation Settings")
        weight_decay = st.select_slider(
            "Weight Decay",
            options=[1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
            help="Higher values prevent overfitting (1e-3 recommended)"
        )
        
        use_early_stopping = st.checkbox("Use Early Stopping", value=True)
        patience = st.slider("Early Stopping Patience", min_value=3, max_value=10, value=8, 
                           help="Lower values stop training sooner to prevent overfitting")
    
    with col2:
        batch_size = st.select_slider(
            "Batch Size",
            options=[16, 32, 64, 128],
            value=32
        )
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0002, 0.0003, 0.0005],  
            value=0.0005,  # Set default to the recommended value
            format_func=lambda x: f"{x:.4f}",
            help="Lower values reduce overfitting (0.0002 recommended for stability)"
        )
        
        use_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True, 
                                  help="Reduces learning rate when validation loss plateaus")
        
        gradient_clipping = st.checkbox("Use Gradient Clipping", value=True,
                              help="Prevents exploding gradients")

        # Add advanced configuration section
        st.markdown("#### 🔧 Advanced Configuration")
        with st.expander("Advanced Training Parameters", expanded=False):
            gradient_clip_max_norm = st.slider(
                "Gradient Clip Max Norm", 
                min_value=0.5, max_value=5.0, value=1.0, step=0.1,
                help="Maximum gradient norm for clipping"
            )
            
            scheduler_factor = st.slider(
                "Scheduler Factor", 
                min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                help="Factor by which learning rate is reduced"
            )
            
            scheduler_patience = st.slider(
                "Scheduler Patience", 
                min_value=2, max_value=10, value=3, step=1,
                help="Epochs to wait before reducing learning rate"
            )
            
            early_stopping_min_epochs = st.slider(
                "Early Stopping Min Epochs", 
                min_value=10, max_value=50, value=20, step=5,
                help="Minimum epochs before early stopping can trigger"
            )
    # Store slider values in session state for batch experiments
    st.session_state.model_config = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        'epochs': epochs,
        'model_type': model_type,
        'use_scheduler': use_scheduler,
        'use_early_stopping': use_early_stopping,
        'gradient_clipping': gradient_clipping,
        'gradient_clip_max_norm': gradient_clip_max_norm,      # ← ADD THESE
        'scheduler_factor': scheduler_factor,                  # ← ADD THESE
        'scheduler_patience': scheduler_patience,              # ← ADD THESE
        'early_stopping_min_epochs': early_stopping_min_epochs  # ← ADD THIS
    }
    
    st.warning("""
    ⚠️ **RESULT VARIABILITY NOTICE**

    Neural network training is inherently stochastic, resulting in:
    - **Different results each training run** due to random weight initialisation
    - **Performance variation** across different CNN architectures and signal methods
    - **Market regime sensitivity** - performance varies between volatile and bull market periods
    - **Classification accuracy** typically ranges 35-40% (better than random 33%)

    **Recommendation:** Run multiple experiments to assess typical performance range rather than relying on single results.
    """)
    
    if st.button("Train Model"):
        # Clear downstream results when training new model
        for key in ['strategy_results', 'benchmark_results', 'random_results', 'strategy_portfolio', 
                   'benchmark_portfolio', 'trades', 'test_dates']:
            if key in st.session_state:
                del st.session_state[key]
        
        try:
            progress_container = st.container()
            
            with progress_container:
                st.markdown("#### Training Progress")
                progress_bar = st.progress(0)
                progress_text = st.empty()
                metrics_text = st.empty()
            
            model = CNNTradingModel(input_shape=(30, 30))
            
            if model_type == "Simple CNN (Recommended)":
                model.build_simple_cnn()
            else:
                model.build_advanced_cnn()
            
            st.markdown("#### Model Summary")
            try:
                sample_input = torch.zeros((1, 1, 30, 30))
                summary_str = str(torch_summary(model.model, input_size=sample_input.shape))
                st.code(summary_str)
            except Exception as e:
                total_params = sum(p.numel() for p in model.model.parameters())
                st.code(f"""
                {model_type} Model
                Total parameters: {total_params:,}
                Input shape: (1, 30, 30)
                Output shape: (3,) - [Hold, Buy, Sell]
                
                Anti-overfitting measures:
                - Dropout: 0.5, 0.7
                - Batch Normalisation: Yes
                - Weight Decay: {weight_decay}
                - Early Stopping: {use_early_stopping}
                """)
            
            def progress_callback(epoch, train_loss, train_acc, val_loss, val_acc):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                
                progress_text.markdown(f"**Epoch {epoch + 1}/{epochs} ({progress*100:.1f}% complete)**")
                
                train_val_gap = abs(train_acc - val_acc)
                if train_val_gap > 0.3:
                    gap_color = "🔴"
                elif train_val_gap > 0.15:
                    gap_color = "🟡"
                else:
                    gap_color = "🟢"
                
                metrics_text.markdown(f"""
                **Current metrics:**
                - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}
                - Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}
                - {gap_color} Overfitting Gap: {train_val_gap:.1%}
                """)
            
            progress_text.markdown(f"**Starting training...**")
            
            history = model.train(
                X_train, y_train, X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
                use_scheduler=use_scheduler,
                use_early_stopping=use_early_stopping,
                use_gradient_clipping=gradient_clipping,                    # ← ADD THESE
                gradient_clip_max_norm=gradient_clip_max_norm,              # ← ADD THESE
                scheduler_factor=scheduler_factor,                          # ← ADD THESE
                scheduler_patience=scheduler_patience,                      # ← ADD THESE
                early_stopping_min_epochs=early_stopping_min_epochs,       # ← ADD THIS
                callback=progress_callback
            )
            
            progress_bar.progress(1.0)
            progress_text.markdown("**Training Complete!**")
            
            st.session_state.model = model
            st.session_state.history = history
            
            st.markdown("#### Training History")
            fig = plot_model_training_history(history)
            st.pyplot(fig)
            
            st.markdown("#### Model Evaluation")
            y_pred_prob = model.predict(X_test)
            st.session_state.predictions = y_pred_prob
            
            metrics = model.evaluate(X_test, y_test)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                acc_color = "🟢" if metrics['accuracy'] > 0.45 else "🟡" if metrics['accuracy'] > 0.35 else "🔴"
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}", help=f"{acc_color} Target: >45%")
            
            with col2:
                if 'class_metrics' in metrics and 'Buy' in metrics['class_metrics']:
                    buy_prec = metrics['class_metrics']['Buy']['precision']
                    st.metric("Buy Precision", f"{buy_prec:.2%}")
                else:
                    st.metric("Buy Precision", "N/A")
            
            with col3:
                if 'class_metrics' in metrics and 'Sell' in metrics['class_metrics']:
                    sell_prec = metrics['class_metrics']['Sell']['precision']
                    st.metric("Sell Precision", f"{sell_prec:.2%}")
                else:
                    st.metric("Sell Precision", "N/A")
            
            with col4:
                if 'class_metrics' in metrics and 'Hold' in metrics['class_metrics']:
                    hold_prec = metrics['class_metrics']['Hold']['precision']
                    st.metric("Hold Precision", f"{hold_prec:.2%}")
                else:
                    st.metric("Hold Precision", "N/A")
            
            # Overfitting analysis
            final_train_acc = history['accuracy'][-1]
            final_val_acc = history['val_accuracy'][-1]
            overfitting_gap = final_train_acc - final_val_acc
            
            if overfitting_gap > 0.3:
                st.error(f"""
                🔴 **High Overfitting Detected**
                - Training Accuracy: {final_train_acc:.2%}
                - Validation Accuracy: {final_val_acc:.2%}
                - Gap: {overfitting_gap:.1%}
                
                **Recommendations:**
                - Increase weight decay to {weight_decay * 10:.0e}
                - Reduce epochs to {epochs // 2}
                - Consider more data augmentation
                """)
            elif overfitting_gap > 0.15:
                st.warning(f"""
                🟡 **Moderate Overfitting**
                - Training Accuracy: {final_train_acc:.2%}
                - Validation Accuracy: {final_val_acc:.2%}
                - Gap: {overfitting_gap:.1%}
                
                **This is acceptable but monitor trading performance**
                """)
            else:
                st.success(f"""
                🟢 **Good Generalisation**
                - Training Accuracy: {final_train_acc:.2%}
                - Validation Accuracy: {final_val_acc:.2%}
                - Gap: {overfitting_gap:.1%}
                
                **Model should perform well in trading**
                """)
            
            # Display evaluation plots
            if y_pred_prob.shape[1] > 1:
                fig = plot_multiclass_evaluation(y_test, y_pred_prob)
                st.pyplot(fig)
            
            model.save(f"{ticker}_improved_model")
            st.success(f"✅ Model trained and saved! Accuracy: {metrics['accuracy']:.1%}")
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            st.exception(e)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_trading_strategy():
    """Display trading strategy page with dark mode compatibility."""
    st.markdown('<div class="sub-header">Trading Strategy</div>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' section.")
        return
    
    if 'predictions' not in st.session_state or st.session_state.predictions is None:
        st.warning("Model predictions are not available. Please train the model first.")
        return
    
    prepared_data = st.session_state.prepared_data
    predictions = st.session_state.predictions
    
    # FIXED: Use Streamlit container instead of custom HTML section
    with st.container():
        st.markdown("### Trading Strategy Configuration")
        
        # FIXED: Use st.info() instead of custom HTML for better dark mode support
        st.info("""
**📊 Experimental Setup:**

Implementation follows Sezer & Ozbayoglu (2018) benchmark methodology:

🏛️ **Exact same data periods:** 2007-2012 and 2012-2017  
📈 **Same stock universe:** Dow 30 components  
🖼️ **30x30 pixel bar chart images** as CNN input  
⚖️ **Three-class prediction:** Buy, Hold, Sell  
⚠️ **Results vary between runs** due to stochastic training
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
            
            transaction_cost = st.slider(
                "Transaction Cost (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                format="%.2f"
            ) / 100
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=50,
                max_value=100,
                value=95,
                step=5,
                help="Maximum percentage of capital to invest in a single position"
            ) / 100
        
        with col2:
            threshold_method = st.radio(
                "Signal Threshold Method",
                options=["Fixed", "Dynamic"],
                index=1,  # Default to Dynamic (optimal configuration)
                help="Dynamic: Adapts thresholds based on signal distribution - provides better risk management and superior performance. Fixed: Uses predetermined confidence thresholds for consistent signal generation."
            )
            
            if threshold_method == "Fixed":
                signal_threshold = st.slider(
                    "Signal Threshold",
                    min_value=0.10,
                    max_value=0.30,
                    value=0.20,
                    step=0.02,
                    help="Lower values = more trades, higher values = fewer trades. The model needs lower thresholds due to low confidence."
                )
            else:
                st.info("Using dynamic thresholds based on signal distribution")
                signal_threshold = None
            
            position_sizing = st.selectbox(
                "Position Sizing Method",
                options=["Fixed Percentage", "Kelly Criterion (Advanced)"],
                index=0,
                help="Fixed percentage is more conservative"
            )
    
    # Show prediction analysis with better dark mode support
    st.markdown("#### 🔍 Prediction Analysis")
    if len(predictions.shape) > 1:
        max_probs = np.max(predictions, axis=1)
        
        # Use metrics layout for better visibility in all modes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Confidence", f"{np.mean(max_probs):.1%}")
        with col2:
            st.metric("Confidence Range", f"{np.min(max_probs):.1%} - {np.max(max_probs):.1%}")
        with col3:
            st.metric("Above 80%", f"{np.sum(max_probs > 0.8) / len(max_probs):.1%}")
        with col4:
            st.metric("Above 50%", f"{np.sum(max_probs > 0.5) / len(max_probs):.1%}")
    
    # FIXED: Use st.warning() instead of custom HTML
    st.warning("""
⚠️ **RESULT VARIABILITY NOTICE**

Neural network training is inherently stochastic, resulting in:
- **Different results each training run** due to random weight initialisation
- **Performance variation** across different CNN architectures and signal methods
- **Market regime sensitivity** - performance varies between volatile and bull market periods
- **Classification accuracy** typically ranges 35-40% (better than random 33%)

**Recommendation:** Run multiple experiments to assess typical performance range rather than relying on single results.
    """)
    
    if st.button("Execute Trading Strategy"):
        # CLEAR ALL PREVIOUS STRATEGY RESULTS WHEN NEW STRATEGY IS EXECUTED
        for key in ['strategy_results', 'benchmark_results', 'random_results', 'strategy_portfolio', 
                   'benchmark_portfolio', 'trades', 'test_dates']:
            if key in st.session_state:
                del st.session_state[key]
        
        with st.spinner("Executing trading strategy..."):
            
            data = st.session_state.data
            ticker = prepared_data['ticker']
            df = data[ticker].copy()
            
            data_loader = DataLoader()
            test_dates = data_loader.get_test_dates(
                data, ticker, window_size=prepared_data['window_size'],
                train_ratio=0.7, val_ratio=0.15
            )
            
            # Align data
            if len(test_dates) > len(predictions):
                test_dates = test_dates[-len(predictions):]
            elif len(test_dates) < len(predictions):
                predictions = predictions[-len(test_dates):]
            
            test_prices = df.loc[test_dates]['Close'].values
            min_length = min(len(predictions), len(test_prices))
            signals = predictions[:min_length]
            prices = test_prices[:min_length]
            aligned_dates = test_dates[:min_length]
            
            # Initialise strategy
            strategy = TradingStrategy(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                max_position_size=max_position_size
            )
            
            # Run CNN strategy with current parameters
            strategy_results = strategy.apply_strategy_optimized(
                prices, signals, aligned_dates,
                ticker=ticker,
                test_period=prepared_data.get('test_period', '2007-2012'),
                fixed_threshold=signal_threshold,
                override_config={
                    'threshold': signal_threshold if signal_threshold else None,
                    'position_size': max_position_size,
                    'method': 'fixed' if threshold_method == "Fixed" else 'dynamic'
                }
            )
            strategy_portfolio = strategy.get_portfolio_values().copy()  # Capture CNN strategy portfolio
            strategy_trades = strategy.get_trades().copy()  # Capture CNN strategy trades

            # Reset and run buy-and-hold
            strategy.reset()
            benchmark_results = strategy.apply_buy_and_hold(prices)
            benchmark_portfolio = strategy.get_portfolio_values().copy()  # Capture buy-and-hold portfolio

            # Reset and run random strategy
            strategy.reset()
            random_results = strategy.apply_random_strategy(prices)
            random_portfolio = strategy.get_portfolio_values().copy()  # Capture random portfolio (if needed)

            # Store all FRESH results in session state
            st.session_state.strategy_results = strategy_results
            st.session_state.benchmark_results = benchmark_results
            st.session_state.random_results = random_results
            st.session_state.strategy_portfolio = strategy_portfolio  # CNN strategy portfolio
            st.session_state.benchmark_portfolio = benchmark_portfolio  # Buy-and-hold portfolio
            st.session_state.trades = strategy_trades  # CNN strategy trades
            st.session_state.test_dates = aligned_dates
            
            # Add a timestamp to track when strategy was executed
            from datetime import datetime
            st.session_state.strategy_executed_at = datetime.now().isoformat()
            st.session_state.strategy_config = {
                'threshold_method': threshold_method,
                'signal_threshold': signal_threshold,
                'initial_capital': initial_capital,
                'transaction_cost': transaction_cost,
                'max_position_size': max_position_size
            }
            
            st.markdown("#### Strategy Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                perf_delta = strategy_results['total_return'] - benchmark_results['total_return']
                st.metric(
                    "Total Return",
                    f"{strategy_results['total_return']:.2%}",
                    f"{perf_delta:.2%} vs B&H",
                    delta_color="normal" if perf_delta > 0 else "inverse"
                )
            
            with col2:
                sharpe_delta = strategy_results['sharpe_ratio'] - benchmark_results['sharpe_ratio']
                st.metric(
                    "Sharpe Ratio",
                    f"{strategy_results['sharpe_ratio']:.2f}",
                    f"{sharpe_delta:.2f} vs B&H",
                    delta_color="normal" if sharpe_delta > 0 else "inverse"
                )
            
            with col3:
                dd_delta = benchmark_results['max_drawdown'] - strategy_results['max_drawdown']
                st.metric(
                    "Max Drawdown",
                    f"{strategy_results['max_drawdown']:.2%}",
                    f"{dd_delta:.2%} vs B&H",
                    delta_color="normal" if dd_delta > 0 else "inverse"
                )
            
            with col4:
                st.metric("Win Rate", f"{strategy_results['win_rate']:.2%}")
            
            # Show which threshold method was used
            threshold_display = f"{signal_threshold:.2f}" if signal_threshold is not None else "Dynamic"
            st.info(f"""
**Strategy Configuration Used:**
- Threshold Method: {threshold_method}
- Threshold Value: {threshold_display}
- Initial Capital: ${initial_capital:,}
- Transaction Cost: {transaction_cost:.2%}
            """)
            
            if strategy_results['total_return'] > benchmark_results['total_return']:
                st.success(f"""
🎉 **Strategy Outperformed!**
- CNN Strategy: {strategy_results['total_return']:.2%}
- Buy & Hold: {benchmark_results['total_return']:.2%}
- Outperformance: {perf_delta:.2%}
                """)
            else:
                if signal_threshold is not None:
                    suggested_threshold = signal_threshold * 0.8
                    threshold_suggestion = f"- Adjust threshold to {suggested_threshold:.2f}"
                else:
                    threshold_suggestion = "- Try fixed threshold of 0.15-0.20"

                st.warning(f"""
📊 **Strategy Underperformed**
- CNN Strategy: {strategy_results['total_return']:.2%}
- Buy & Hold: {benchmark_results['total_return']:.2%}
- Underperformance: {perf_delta:.2%}

**Possible improvements:**
{threshold_suggestion}
- Increase training epochs
- Add more regularisation
                """)
            
            st.markdown("#### Portfolio Performance")
            portfolio_values = st.session_state.strategy_portfolio
            benchmark_values = st.session_state.benchmark_portfolio
            
            fig = plot_interactive_portfolio(
                portfolio_values, benchmark_values,
                aligned_dates, st.session_state.trades
            )
            st.plotly_chart(fig, use_container_width=True)
            
            trades_df = st.session_state.trades
            if len(trades_df) > 0:
                st.markdown("#### Trading Activity")
                st.dataframe(trades_df.tail(10))
                
                buy_trades = len(trades_df[trades_df['type'] == 'BUY'])
                sell_trades = len(trades_df[trades_df['type'].isin(['SELL', 'SELL_FINAL'])])
                st.info(f"**Trading Summary:** {buy_trades} buys, {sell_trades} sells")
            else:
                st.info("No trades were executed (all signals were Hold)")
    
    # Show current strategy configuration if results exist
    if 'strategy_results' in st.session_state and 'strategy_config' in st.session_state:
        st.markdown("#### Current Strategy Status")
        config = st.session_state.strategy_config
        executed_at = st.session_state.get('strategy_executed_at', 'Unknown')
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
**Last Executed:** {executed_at[:19] if executed_at != 'Unknown' else 'Unknown'}

**Configuration:**
- Method: {config['threshold_method']}
- Threshold: {config['signal_threshold'] if config['signal_threshold'] is not None else 'Dynamic'}
            """)
        
        with col2:
            st.info(f"""
**Parameters:**
- Capital: ${config['initial_capital']:,}
- Transaction Cost: {config['transaction_cost']:.2%}
- Max Position: {config['max_position_size']:.0%}
            """)

def display_performance_analysis():
    """Display performance analysis page with corrected metrics."""
    st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
    
    # Validate that we have current results
    if 'strategy_results' not in st.session_state or st.session_state.strategy_results is None:
        st.warning("Please execute a trading strategy first in the 'Trading Strategy' section.")
        return
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' section.")
        return
    
    if 'prepared_data' not in st.session_state or st.session_state.prepared_data is None:
        st.warning("Please prepare data first in the 'Data Preparation' section.")
        return
    
    # Get current results - these should be fresh from the latest execution
    strategy_results = st.session_state.strategy_results
    benchmark_results = st.session_state.benchmark_results
    
    ticker = st.session_state.prepared_data['ticker'] if 'prepared_data' in st.session_state else "selected stock"
    test_period = st.session_state.prepared_data.get('test_period', 'Unknown') if 'prepared_data' in st.session_state else 'Unknown'
    
    model_accuracy = "N/A"
    if 'model' in st.session_state and st.session_state.model is not None:
        try:
            if 'prepared_data' in st.session_state:
                prepared_data = st.session_state.prepared_data
                model_metrics = st.session_state.model.evaluate(
                    prepared_data['X_test'], 
                    prepared_data['y_test']
                )
                model_accuracy = f"{model_metrics['accuracy']:.1%}"
        except Exception as e:
            model_accuracy = "N/A"
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f"### Performance Report - {ticker} ({test_period})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Strategy Performance")
        st.metric("Classification Accuracy", model_accuracy)
        st.metric("Total Return", f"{strategy_results['total_return']:.2%}")
        st.metric("Annualised Return", f"{strategy_results['annualised_return']:.2%}")
        st.metric("Sharpe Ratio", f"{strategy_results['sharpe_ratio']:.3f}")
        st.metric("Maximum Drawdown", f"{strategy_results['max_drawdown']:.2%}")
        st.metric("Win Rate", f"{strategy_results['win_rate']:.2%}")
        st.metric("Number of Trades", f"{strategy_results['number_of_trades']}")
    
    with col2:
        st.markdown("#### 🎯 Buy & Hold Benchmark")
        if benchmark_results:
            st.metric("Total Return", f"{benchmark_results['total_return']:.2%}")
            st.metric("Annualised Return", f"{benchmark_results['annualised_return']:.2%}")
            st.metric("Sharpe Ratio", f"{benchmark_results['sharpe_ratio']:.3f}")
            st.metric("Maximum Drawdown", f"{benchmark_results['max_drawdown']:.2%}")
    
    # Benchmark paper comparison
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📋 Benchmark Paper Comparison")
    
    success_rate_single = "100.0%" if (benchmark_results and strategy_results['total_return'] > benchmark_results['total_return']) else "0.0%"
    
    if test_period == "2007-2012":
        paper_accuracy = "55.2%"
        paper_cnn_return = "7.20%"
        paper_bah_return = "5.86%"
        paper_cnn_drawdown = "27.2%"
        paper_bah_drawdown = "35.2%"
        paper_success_rate = "79.3%"  
    else:
        paper_accuracy = "53.4%"      
        paper_cnn_return = "5.84%"
        paper_bah_return = "13.25%"
        paper_cnn_drawdown = "19.5%"
        paper_bah_drawdown = "6.97%"
        paper_success_rate = "3.4%"
    
    comparison_data = {
        'Metric': [
            'Classification Accuracy',
            'CNN Strategy Annualised Return', 
            'Buy & Hold Annualised Return',
            'CNN Strategy Max Drawdown',
            'Buy & Hold Max Drawdown',
            'Success Rate vs Buy & Hold',
            'Number of Stocks Tested'
        ],
        'Model Results': [
            model_accuracy,
            f"{strategy_results['annualised_return']:.2%}",
            f"{benchmark_results['annualised_return']:.2%}" if benchmark_results else 'N/A',
            f"{strategy_results['max_drawdown']:.2%}",
            f"{benchmark_results['max_drawdown']:.2%}" if benchmark_results else 'N/A',
            success_rate_single,
            '1 (Single Stock Analysis)'
        ],
        'Sezer & Ozbayoglu (2018)': [
            paper_accuracy,
            paper_cnn_return,
            paper_bah_return,
            paper_cnn_drawdown,
            paper_bah_drawdown,
            paper_success_rate,
            '29 (Dow 30 Analysis)'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🔍 Analysis of Results")
    
    if benchmark_results:
        outperformance = strategy_results['total_return'] - benchmark_results['total_return']
        
        if outperformance < 0:
            st.error(f"""
            ❌ **Strategy Underperformed**: The CNN strategy underperformed buy-and-hold by 
            **{-outperformance:.2%}** during the {test_period} period.
            
            Check the detailed metrics below to understand performance drivers.
            """)
        else:
            st.success(f"""
            ✅ **Strategy Outperformed**: The CNN strategy outperformed buy-and-hold by 
            **{outperformance:.2%}** during the {test_period} period.
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_batch_experiment_page():
    """Display the batch experiment page - FIXED to match benchmark paper exactly"""
    st.markdown('<div class="sub-header">Comprehensive Benchmark Experiment</div>', unsafe_allow_html=True)
    
    # Display current model configuration
    config = st.session_state.get('model_config', None)
    
    if config is None:
        st.warning("""
        ⚠️ **No Model Configuration Found**
        
        Please visit the **"Model Training (Single Stock)"** section first to set your model parameters.
        The batch experiment will use those same settings for consistency.
        """)
        return
    
    # Show current configuration
    st.info(f"""
    **Current Model Configuration** (set in Model Training section):
    - Architecture: {config['model_type']}
    - Batch Size: {config['batch_size']}
    - Learning Rate: {config['learning_rate']:.4f}
    - Weight Decay: {config['weight_decay']:.0e}
    - Early Stopping Patience: {config['patience']}
    - Training Epochs: {config['epochs']}
    
    💡 *To modify these settings, go to "Model Training (Single Stock)" → "Model Configuration"*
    """)
    
    st.markdown("""
    <div class="info-box">
        <strong>Benchmark Paper Methodology:</strong> This experiment tests all 30 Dow Jones Industrial Average stocks 
        using the exact same methodology as Sezer & Ozbayoglu (2018) for direct comparison.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_period = st.selectbox(
            "Select Test Period",
            options=["2007-2012", "2012-2017"],
            help="Choose the benchmark test period"
        )
        
        # CHECK IF TEST PERIOD CHANGED AND CLEAR PREVIOUS RESULTS
        if 'previous_test_period' not in st.session_state:
            st.session_state.previous_test_period = test_period
        
        if st.session_state.previous_test_period != test_period:
            # Clear all previous results when test period changes
            st.session_state.show_current_results = False
            if 'batch_results' in st.session_state:
                del st.session_state.batch_results
            if 'batch_successful_results' in st.session_state:
                del st.session_state.batch_successful_results
            if 'batch_test_period' in st.session_state:
                del st.session_state.batch_test_period
            if 'batch_unique_key' in st.session_state:
                del st.session_state.batch_unique_key
            
            # Update the stored test period
            st.session_state.previous_test_period = test_period
            
            # Force a rerun to clear the display
            st.rerun()
        
        # REMOVED SLIDER - Use fixed paper methodology
        st.success("""
        **📋 Paper-Accurate Testing:**
        - Testing ALL 30 Dow Jones stocks
        - Exact same methodology as benchmark paper
        - Direct performance comparison possible
        """)
        
        quick_run = st.checkbox(
            "Quick Run (Reduced Epochs)", 
            value=False,
            help="Use fewer epochs for faster testing (30 vs configured epochs)"
        )
    
    with col2:
        dow30_stocks = [
            "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
            "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
            "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
        ]
        
        # FIXED: Show all stocks, allow user to exclude specific ones if needed
        stock_selection = st.multiselect(
            "Stocks to Test (all 30 Dow stocks selected by default)",
            options=dow30_stocks,
            default=dow30_stocks,  # CHANGED: All 30 stocks like the paper!
            help="All Dow 30 stocks are selected by default to match the benchmark paper. You can exclude specific stocks if needed."
        )
        
        # Show count
        st.info(f"**Selected: {len(stock_selection)}/30 stocks**")
        
        if len(stock_selection) < 20:
            st.warning("⚠️ Testing fewer than 20 stocks may not provide statistically significant results.")
    
    # Show paper comparison info (only for current selection)
    st.markdown("### 📋 Benchmark Paper Comparison")
    if test_period == "2007-2012":
        st.info("""
        **Sezer & Ozbayoglu (2018) Benchmark Results for 2007-2012:**
        - Average Classification Accuracy: 55.2%
        - CNN Strategy Return: 7.20%
        - Buy & Hold Return: 5.86%
        - Success Rate: 79.3% (23/29 stocks outperformed)
        - **Stocks Tested: 29 Dow 30 stocks** 
        """)
    else:
        st.info("""
        **Sezer & Ozbayoglu (2018) Results for 2012-2017:**
        - Average Classification Accuracy: 53.4%
        - CNN Strategy Return: 5.84%
        - Buy & Hold Return: 13.25%
        - Trading Success Rate: 53.4% (profitable individual trades)
        - Stock Outperformance: ~3.4% (1-2/29 stocks beat B&H)
        - **Stocks Tested: 29 Dow stocks** (strong bull market period)
        """)
    
    # Run experiment button
    if st.button("🚀 Run Comprehensive Benchmark Experiment"):
        if len(stock_selection) < 10:
            st.warning("⚠️ Consider testing at least 10 stocks for meaningful statistical comparison.")
        
        # Clear previous results flag
        st.session_state.show_current_results = False
        
        with st.spinner("Running comprehensive benchmark experiment..."):
            summary = run_batch_experiment_direct(
                test_period=test_period,
                selected_stocks=stock_selection,  # Use the selected stocks
                max_stocks=len(stock_selection),  # Use actual count
                quick_run=quick_run
            )
            
            if summary:
                st.success("✅ Comprehensive benchmark experiment completed successfully!")
    
    # Show current results ONLY if:
    # 1. Flag is set to show results
    # 2. Results exist
    # 3. Results are for the CURRENT test period (not previous)
    if (st.session_state.get('show_current_results', False) and 
        'batch_results' in st.session_state and 
        st.session_state.batch_results is not None and
        st.session_state.get('batch_test_period') == test_period):
        
        st.markdown("---")  # Visual separator
        
        # Get stored results
        summary = st.session_state.batch_results
        successful_results = st.session_state.get('batch_successful_results', [])
        test_period_used = st.session_state.get('batch_test_period', 'Unknown')
        unique_key = st.session_state.get('batch_unique_key', 'default')
        
        # Show when results were generated
        st.info(f"📊 **Current Results** | Test Period: {test_period_used} | Stocks: {len(successful_results)}")
        
        # Display the results with unique key
        if successful_results:
            display_batch_results_fixed(summary, successful_results, test_period_used, unique_key)
        else:
            st.warning("Detailed results not available. Please run a new experiment.")

def clear_batch_results():
    """Helper function to manually clear all batch results from session state"""
    keys_to_clear = [
        'batch_results', 
        'batch_successful_results', 
        'batch_test_period', 
        'batch_unique_key', 
        'show_current_results',
        'previous_test_period'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def display_comprehensive_benchmark_comparison(summary, test_period):
    """Display comprehensive comparison with benchmark paper."""
    st.markdown("### 📊 Comprehensive Benchmark Analysis")
    
    benchmark_paper = get_benchmark_paper_results_fixed(test_period)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Model Results")
        st.metric("Stocks Tested", f"{summary['num_stocks']}")
        st.metric("Avg Classification Accuracy", f"{summary['avg_accuracy']:.1%}")
        st.metric("Success Rate vs B&H", f"{summary['success_rate']:.1%}")
        st.metric("Avg Strategy Return", f"{summary['avg_strategy_return']:.2%}")
        st.metric("Avg B&H Return", f"{summary['avg_benchmark_return']:.2%}")
    
    with col2:
        st.markdown("#### 📋 Benchmark Paper")
        st.metric("Stocks Tested", "29")
        st.metric("Avg Classification Accuracy", f"{benchmark_paper['classification_accuracy']:.1%}")
        st.metric("Success Rate vs B&H", f"{benchmark_paper['success_rate']:.1%}")
        st.metric("CNN Strategy Return", f"{benchmark_paper['cnn_return']:.2%}")
        st.metric("Buy & Hold Return", f"{benchmark_paper['bah_return']:.2%}")
    
    with col3:
        st.markdown("#### 📈 Comparison")
        acc_diff = summary['avg_accuracy'] - benchmark_paper['classification_accuracy']
        success_diff = summary['success_rate'] - benchmark_paper['success_rate']
        return_diff = summary['avg_strategy_return'] - benchmark_paper['cnn_return']
        
        st.metric("Accuracy Difference", f"{acc_diff:+.1%}")
        st.metric("Success Rate Difference", f"{success_diff:+.1%}")
        st.metric("Return Difference", f"{return_diff:+.2%}")
        
        if success_diff > 0 and return_diff > 0:
            st.success("🎉 Outperformed benchmark!")
        elif success_diff > 0 or return_diff > 0:
            st.info("📊 Mixed results vs benchmark")
        else:
            st.warning("📉 Underperformed benchmark")

def check_data_sufficiency(splits, ticker, window_size=30, prediction_horizon=5):
    """Check if ticker has sufficient data for analysis"""
    train_data = splits['train_data']
    test_data = splits['test_data']
    
    min_train_samples = window_size + prediction_horizon + 100  # Need reasonable sample size
    min_test_samples = prediction_horizon + 20  # Minimum test samples
    
    issues = []
    
    if len(train_data) < min_train_samples:
        issues.append(f"Insufficient training data: {len(train_data)} rows, need {min_train_samples}")
    
    if len(test_data) < min_test_samples:
        issues.append(f"Insufficient test data: {len(test_data)} rows, need {min_test_samples}")
        
    # Check for excessive NaN values
    train_nan_pct = train_data.isnull().sum().sum() / (len(train_data) * len(train_data.columns))
    if train_nan_pct > 0.1:  # More than 10% NaN
        issues.append(f"Excessive missing data: {train_nan_pct:.1%}")
    
    return issues

def check_data_sufficiency(splits, ticker, window_size=30, prediction_horizon=5):
    """Check if ticker has sufficient data for analysis"""
    train_data = splits['train_data']
    test_data = splits['test_data']
    
    min_train_samples = window_size + prediction_horizon + 100  # Need reasonable sample size
    min_test_samples = prediction_horizon + 20  # Minimum test samples
    
    issues = []
    
    if len(train_data) < min_train_samples:
        issues.append(f"Insufficient training data: {len(train_data)} rows, need {min_train_samples}")
    
    if len(test_data) < min_test_samples:
        issues.append(f"Insufficient test data: {len(test_data)} rows, need {min_test_samples}")
        
    # Check for excessive NaN values
    train_nan_pct = train_data.isnull().sum().sum() / (len(train_data) * len(train_data.columns))
    if train_nan_pct > 0.1:  # More than 10% NaN
        issues.append(f"Excessive missing data: {train_nan_pct:.1%}")
    
    return issues

def run_batch_experiment_direct(test_period=None, cnn_architecture=None, signal_method=None, 
                              epochs=None, batch_size=None, early_stopping_patience=None, 
                              quick_run=False, **kwargs):
    """Run batch experiment using Model Training settings across all 30 Dow stocks - EODHD version"""
    
    try:
        # Import the working modules from the old version
        from utils.data_loader import DataLoader
        from utils.feature_engineering import FeatureEngineer
        from utils.model import CNNTradingModel
        from utils.trading_strategy import TradingStrategy
        import traceback
        from datetime import datetime
        
        # Check EODHD configuration first (from old working version)
        try:
            from utils.eodhdapi import get_client
            client = get_client()
            st.info("✅ Using EODHD API for batch experiment")
        except Exception as e:
            st.error(f"❌ EODHD API Error: {str(e)}")
            st.stop()
            return None
        
        # Get the 30 Dow stocks (same as new version)
        dow_30_stocks = [
            'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
            'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
            'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM'
        ]
        
        # Get model configuration from Model Training page (improved from new version)
        def get_model_config():
            config = {}
            
            # Architecture
            for key in ['cnn_architecture', 'model_architecture', 'architecture', 'selected_architecture']:
                if key in st.session_state:
                    config['architecture'] = st.session_state[key]
                    break
            else:
                config['architecture'] = 'SimpleCNN'  # Default
                
            # Epochs
            for key in ['epochs', 'training_epochs', 'max_epochs', 'num_epochs']:
                if key in st.session_state:
                    config['epochs'] = st.session_state[key]
                    break
            else:
                config['epochs'] = 100  # Default
                
            # Signal method
            for key in ['signal_method', 'threshold_method', 'signal_threshold_method']:
                if key in st.session_state:
                    config['signal_method'] = st.session_state[key]
                    break
            else:
                config['signal_method'] = 'Dynamic'  # Default
                
            # Other parameters
            config['batch_size'] = st.session_state.get('batch_size', 32)
            config['early_stopping_patience'] = st.session_state.get('early_stopping_patience', 5)
            config['test_period'] = test_period or '2007-2012'
            
            return config
        
        # Get configuration (from new version)
        config = get_model_config()
        
        # Override with any passed parameters
        if cnn_architecture: config['architecture'] = cnn_architecture
        if epochs: config['epochs'] = epochs
        if signal_method: config['signal_method'] = signal_method
        if batch_size: config['batch_size'] = batch_size
        if early_stopping_patience: config['early_stopping_patience'] = early_stopping_patience
        
        # Handle quick run
        selected_stocks = dow_30_stocks.copy()
        if quick_run:
            config['epochs'] = min(config['epochs'], 30)  # Reasonable for quick run
            selected_stocks = selected_stocks[:3]  # First 3 stocks only
            st.info(f"🏃‍♂️ Quick Run: Testing {len(selected_stocks)} stocks with {config['epochs']} epochs")
        
        # Display configuration (from new version)
        st.write("## 🔧 Experiment Configuration")
        st.write(f"**Architecture:** {config['architecture']}")
        st.write(f"**Epochs:** {config['epochs']}")
        st.write(f"**Signal Method:** {config['signal_method']}")
        st.write(f"**Test Period:** {config['test_period']}")
        st.write(f"**Stocks to Process:** {len(selected_stocks)}")
        
        # Initialize components (from old working version)
        data_loader = DataLoader()
        feature_engineer = FeatureEngineer()
        
        # Download data first (from old working version)
        st.info(f"Starting batch experiment for {config['test_period']} with {len(selected_stocks)} stocks using EODHD")
        
        with st.spinner("Downloading benchmark data from EODHD..."):
            # EODHD allows larger batch sizes
            data = data_loader.download_benchmark_data(force_download=False, batch_size=10)
        
        # Data sufficiency check (improved from new version)
        def check_data_sufficiency(splits, ticker, window_size=30, prediction_horizon=5):
            train_data = splits['train_data']
            test_data = splits['test_data']
            
            min_train_samples = window_size + prediction_horizon + 100
            min_test_samples = prediction_horizon + 20
            
            issues = []
            if len(train_data) < min_train_samples:
                issues.append(f"Insufficient training data: {len(train_data)} rows, need {min_train_samples}")
            if len(test_data) < min_test_samples:
                issues.append(f"Insufficient test data: {len(test_data)} rows, need {min_test_samples}")
                
            return issues
        
        # Track results (from both versions)
        results = {}
        successful_results = []
        failed_tickers = []
        
        # Progress tracking (from old version)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each ticker (combined approach)
        for i, ticker in enumerate(selected_stocks):
            status_text.text(f"Processing {ticker} ({i+1}/{len(selected_stocks)})")
            
            try:
                # Check if ticker exists in downloaded data (from old version)
                if ticker not in data:
                    st.warning(f"⚠️ Skipping {ticker} - no data available")
                    failed_tickers.append({
                        'ticker': ticker,
                        'reason': 'No Data Available',
                        'issues': ['Ticker not found in EODHD data']
                    })
                    continue
                
                # Get data splits (from old version)
                splits = data_loader.get_benchmark_splits(data, ticker, config['test_period'])
                
                # Check data sufficiency (from new version)
                data_issues = check_data_sufficiency(splits, ticker)
                if data_issues:
                    st.warning(f"⚠️ Skipping {ticker}: {', '.join(data_issues)}")
                    failed_tickers.append({
                        'ticker': ticker,
                        'reason': 'Insufficient Data',
                        'issues': data_issues
                    })
                    continue
                
                # Prepare features (from old version)
                prepared_data = feature_engineer.prepare_benchmark_features(
                    splits['train_data'], splits['test_data'], ticker
                )
                
                # Additional check after feature engineering (from new version)
                if prepared_data['X_train'].shape[0] == 0:
                    st.warning(f"⚠️ Skipping {ticker}: No training samples after feature engineering")
                    failed_tickers.append({
                        'ticker': ticker,
                        'reason': 'No Training Samples',
                        'issues': ['Feature engineering produced empty dataset']
                    })
                    continue
                
                # Create and train model (from old version, but with config)
                model = CNNTradingModel()
                
                # Use architecture from config
                if config['architecture'] == 'SimpleCNN':
                    model.build_simple_cnn()
                else:
                    model.build_advanced_cnn()  # Assuming this method exists
                
                # Training callback (from old version)
                def simple_callback(epoch, train_loss, train_acc, val_loss, val_acc):
                    if epoch % 10 == 0:
                        status_text.text(f"Processing {ticker} - Epoch {epoch+1}/{config['epochs']}")
                
                # Train model (from old version with config epochs)
                history = model.train(
                    prepared_data['X_train'], prepared_data['y_train'],
                    prepared_data['X_val'], prepared_data['y_val'],
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    learning_rate=0.001,
                    callback=simple_callback
                )
                
                # Evaluate model (from old version)
                y_pred_prob = model.predict(prepared_data['X_test'])
                metrics = model.evaluate(prepared_data['X_test'], prepared_data['y_test'])
                
                # Trading strategy (from old version)
                strategy = TradingStrategy(initial_capital=10000, transaction_cost=0.001)
                
                test_data = splits['test_data']
                test_prices = test_data['Close'].values
                test_dates = test_data.index
                
                min_len = min(len(y_pred_prob), len(test_prices))
                signals = y_pred_prob[:min_len]
                prices = test_prices[:min_len]
                dates = test_dates[:min_len]
                
                strategy_results = strategy.apply_strategy_optimized(
                    prices, signals, dates,
                    ticker=ticker,
                    test_period=config['test_period']
                )
                
                strategy.reset()
                benchmark_results = strategy.apply_buy_and_hold(prices)
                
                outperformed = strategy_results['total_return'] > benchmark_results['total_return']
                
                # Store results (from old version)
                result = {
                    'ticker': ticker,
                    'classification_accuracy': metrics['accuracy'],
                    'strategy_results': strategy_results,
                    'benchmark_results': benchmark_results,
                    'model_metrics': metrics,
                    'test_period': config['test_period'],
                    'outperformed_benchmark': outperformed
                }
                
                results[ticker] = result
                successful_results.append(result)
                
                # Display result (from old version)
                status_icon = "✅" if outperformed else "❌"
                st.write(f"{status_icon} **{ticker}**: Accuracy={metrics['accuracy']:.1%}, "
                        f"CNN Return={strategy_results['total_return']:.1%}, "
                        f"B&H Return={benchmark_results['total_return']:.1%}, "
                        f"Outperformed: {'Yes' if outperformed else 'No'}")
                
            except Exception as e:
                st.error(f"❌ Error processing {ticker}: {str(e)}")
                failed_tickers.append({
                    'ticker': ticker,
                    'reason': 'Processing Error',
                    'error': str(e)
                })
            
            # Update progress
            progress_bar.progress((i + 1) / len(selected_stocks))
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display summary (improved from both versions)
        st.write("## 📋 Experiment Summary")
        successful_count = len(successful_results)
        failed_count = len(failed_tickers)
        st.write(f"✅ **Successful:** {successful_count} tickers")
        st.write(f"❌ **Failed:** {failed_count} tickers")
        
        if successful_results:
            successful_tickers = [r['ticker'] for r in successful_results]
            st.write(f"**Successful:** {', '.join(successful_tickers)}")
        
        if failed_tickers:
            st.write("### Failed Tickers:")
            for failed in failed_tickers:
                st.write(f"• **{failed['ticker']}**: {failed['reason']}")
        
        # Generate and store results (from old version)
        if successful_results:
            summary = generate_batch_summary_direct(successful_results, config['test_period'])
            
            # Store results in session state with unique timestamp (from old version)
            unique_key = f"{config['test_period']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            st.session_state.batch_results = summary
            st.session_state.batch_successful_results = successful_results
            st.session_state.batch_test_period = config['test_period']
            st.session_state.batch_unique_key = unique_key
            st.session_state.show_current_results = True
            
            st.success("✅ Batch experiment completed using EODHD data!")
            
            return summary
        else:
            st.error("No successful results generated")
            return None
            
    except Exception as e:
        st.error(f"Error in batch experiment: {str(e)}")
        st.exception(e)
        return None

def display_batch_results_fixed(summary, successful_results, test_period, unique_key):
    """Display results with fixed duplicate key issues"""
    
    st.markdown("### 📊 Comprehensive Experiment Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stocks Tested", summary['num_stocks'])
    with col2:
        st.metric("Average Accuracy", f"{summary['avg_accuracy']:.1%}")
    with col3:
        st.metric("Success Rate vs B&H", f"{summary['success_rate']:.1%}")
    with col4:
        st.metric("Average Outperformance", f"{summary['avg_outperformance']:.2%}")
    
    # Individual stock results table
    st.markdown("### 📋 Individual Stock Results")
    
    # Create results table
    results_data = []
    for result in successful_results:
        results_data.append({
            'Ticker': result['ticker'],
            'Accuracy': f"{result['classification_accuracy']:.1%}",
            'Strategy Return': f"{result['strategy_results']['total_return']:.1%}",
            'B&H Return': f"{result['benchmark_results']['total_return']:.1%}",
            'Outperformance': f"{(result['strategy_results']['total_return'] - result['benchmark_results']['total_return']):.1%}",
            'Max Drawdown': f"{result['strategy_results']['max_drawdown']:.1%}",
            'Sharpe Ratio': f"{result['strategy_results']['sharpe_ratio']:.2f}",
            'Trades': result['strategy_results']['number_of_trades'],
            'Outperformed': "✅" if result['outperformed_benchmark'] else "❌"
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Benchmark comparison
    display_comprehensive_benchmark_comparison(summary, test_period)
    
    # CSV Download
    st.markdown("### 📥 Download Results")
    
    # Create CSV data in memory
    summary_data = []
    for result in successful_results:
        summary_data.append({
            'Ticker': result['ticker'],
            'Classification_Accuracy': result['classification_accuracy'],
            'Strategy_Return': result['strategy_results']['total_return'],
            'BuyHold_Return': result['benchmark_results']['total_return'],
            'Outperformance': result['strategy_results']['total_return'] - 
                            result['benchmark_results']['total_return'],
            'Strategy_Drawdown': result['strategy_results']['max_drawdown'],
            'BuyHold_Drawdown': result['benchmark_results']['max_drawdown'],
            'Sharpe_Ratio': result['strategy_results']['sharpe_ratio'],
            'Num_Trades': result['strategy_results']['number_of_trades'],
            'Win_Rate': result['strategy_results']['win_rate'],
            'Outperformed': result['outperformed_benchmark']
        })
    
    if summary_data:
        # Convert to CSV string in memory
        summary_df = pd.DataFrame(summary_data)
        csv_string = summary_df.to_csv(index=False)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"benchmark_results_{test_period}_{timestamp}.csv"
        
        # Single CSV download button with unique key
        st.download_button(
            label="📥 Download CSV Results",
            data=csv_string,
            file_name=csv_filename,
            mime='text/csv',
            key=f"csv_download_{unique_key}",  # Use the unique key from session state
            help="Download complete results as CSV file",
            use_container_width=False
        )
        
        st.success("✅ CSV file ready for download! Results will remain visible after downloading.")


def display_batch_results(summary, successful_results, test_period):
    """Display results that persist after download"""
    
    st.markdown("### 📊 Comprehensive Experiment Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stocks Tested", summary['num_stocks'])
    with col2:
        st.metric("Average Accuracy", f"{summary['avg_accuracy']:.1%}")
    with col3:
        st.metric("Success Rate vs B&H", f"{summary['success_rate']:.1%}")
    with col4:
        st.metric("Average Outperformance", f"{summary['avg_outperformance']:.2%}")
    
    # Individual stock results table
    st.markdown("### 📋 Individual Stock Results")
    
    # Create results table
    results_data = []
    for result in successful_results:
        results_data.append({
            'Ticker': result['ticker'],
            'Accuracy': f"{result['classification_accuracy']:.1%}",
            'Strategy Return': f"{result['strategy_results']['total_return']:.1%}",
            'B&H Return': f"{result['benchmark_results']['total_return']:.1%}",
            'Outperformance': f"{(result['strategy_results']['total_return'] - result['benchmark_results']['total_return']):.1%}",
            'Max Drawdown': f"{result['strategy_results']['max_drawdown']:.1%}",
            'Sharpe Ratio': f"{result['strategy_results']['sharpe_ratio']:.2f}",
            'Trades': result['strategy_results']['number_of_trades'],
            'Outperformed': "✅" if result['outperformed_benchmark'] else "❌"
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Benchmark comparison
    display_comprehensive_benchmark_comparison(summary, test_period)
    
    # Download section
    handle_csv_download(successful_results, test_period)

def handle_csv_download(successful_results, test_period):
    """Handle CSV creation and download without causing refresh"""
    
    st.markdown("### 📥 Download Results")
    
    # Create CSV data in memory (don't write to file)
    summary_data = []
    for result in successful_results:
        summary_data.append({
            'Ticker': result['ticker'],
            'Classification_Accuracy': result['classification_accuracy'],
            'Strategy_Return': result['strategy_results']['total_return'],
            'BuyHold_Return': result['benchmark_results']['total_return'],
            'Outperformance': result['strategy_results']['total_return'] - 
                            result['benchmark_results']['total_return'],
            'Strategy_Drawdown': result['strategy_results']['max_drawdown'],
            'BuyHold_Drawdown': result['benchmark_results']['max_drawdown'],
            'Sharpe_Ratio': result['strategy_results']['sharpe_ratio'],
            'Num_Trades': result['strategy_results']['number_of_trades'],
            'Win_Rate': result['strategy_results']['win_rate'],
            'Outperformed': result['outperformed_benchmark']
        })
    
    if summary_data:
        # Convert to CSV string in memory (no file writing)
        summary_df = pd.DataFrame(summary_data)
        csv_string = summary_df.to_csv(index=False)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"benchmark_results_{test_period}_{timestamp}.csv"
        
        # CSV download button
        st.download_button(
            label="📥 Download CSV Results",
            data=csv_string,
            file_name=csv_filename,
            mime='text/csv',
            key=f"download_csv_{test_period}_{timestamp}",
            help="Download summary results as CSV file",
            use_container_width=True
        )
        
        st.success("✅ CSV file ready for download! Results will remain visible after downloading.")


def generate_batch_summary_direct(results, test_period):
    """Generate summary from batch results."""
    accuracies = [r['classification_accuracy'] for r in results]
    strategy_returns = [r['strategy_results']['total_return'] for r in results]
    benchmark_returns = [r['benchmark_results']['total_return'] for r in results]
    strategy_drawdowns = [r['strategy_results']['max_drawdown'] for r in results]
    benchmark_drawdowns = [r['benchmark_results']['max_drawdown'] for r in results]
    
    outperformances = [r['outperformed_benchmark'] for r in results]
    success_count = sum(outperformances)
    success_rate = success_count / len(results)
    
    return_differences = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
    avg_outperformance = np.mean(return_differences)
    
    return {
        'test_period': test_period,
        'num_stocks': len(results),
        'avg_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'avg_strategy_return': np.mean(strategy_returns),
        'avg_benchmark_return': np.mean(benchmark_returns),
        'avg_strategy_drawdown': np.mean(strategy_drawdowns),
        'avg_benchmark_drawdown': np.mean(benchmark_drawdowns),
        'success_rate': success_rate,
        'success_count': success_count,
        'avg_outperformance': avg_outperformance,
        'individual_results': results
    }

def get_benchmark_paper_results_fixed(test_period):
    """Get benchmark paper results for comparison."""
    if test_period == "2007-2012":
        return {
            'classification_accuracy': 0.552,
            'cnn_return': 0.072,
            'bah_return': 0.0586,
            'cnn_drawdown': 0.272,
            'bah_drawdown': 0.352,
            'success_rate': 0.767
        }
    else:
        return {
            'classification_accuracy': 0.538,
            'cnn_return': 0.0584,
            'bah_return': 0.1325,
            'cnn_drawdown': 0.195,
            'bah_drawdown': 0.0697,
            'success_rate': 0.448
        }

def plot_multiclass_evaluation(y_true, y_pred_prob, figsize=(16, 12)):
    """Plot model evaluation metrics for multi-class classification."""
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Hold', 'Buy', 'Sell']
    num_classes = cm.shape[0]
    used_class_names = class_names[:num_classes]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 0], 
                xticklabels=used_class_names, yticklabels=used_class_names)
    
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_title('Confusion Matrix')
    
    # Plot ROC curves for each class (one-vs-rest)
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    
    for i, class_name in enumerate(used_class_names):
        if i < y_pred_prob.shape[1]:
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_prob[:, i]
            
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                axes[0, 1].plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                continue
    
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves (One-vs-Rest)')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True)
    
    # Plot class distribution
    class_counts = np.bincount(y_true, minlength=len(used_class_names))
    bars = axes[1, 0].bar(range(len(used_class_names)), class_counts[:len(used_class_names)])
    
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Class Distribution')
    axes[1, 0].set_xticks(range(len(used_class_names)))
    axes[1, 0].set_xticklabels(used_class_names)
    axes[1, 0].grid(True, axis='y')
    
    # Plot prediction distribution
    pred_class_counts = np.bincount(y_pred, minlength=len(used_class_names))
    bars = axes[1, 1].bar(range(len(used_class_names)), pred_class_counts[:len(used_class_names)])
    
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    axes[1, 1].set_xlabel('Predicted Class')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].set_xticks(range(len(used_class_names)))
    axes[1, 1].set_xticklabels(used_class_names)
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    return fig

# Run the app
if __name__ == "__main__":
    main()