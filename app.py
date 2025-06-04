"""
Intelligent Algorithmic Trading System

This Streamlit application implements an intelligent algorithmic trading system using
Convolutional Neural Networks (CNNs) based on the approach from Sezer & Ozbayoglu (2018).

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
from datetime import datetime
import sys
import warnings



# Set page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Intelligent Algorithmic Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Fix for PyTorch-Streamlit compatibility issue - AFTER set_page_config
try:
    import streamlit.watcher.local_sources_watcher
    
    # Store the original function
    _original_get_module_paths = streamlit.watcher.local_sources_watcher.get_module_paths
    
    def patched_get_module_paths(module):
        """Patched version that ignores torch._classes and other problematic modules"""
        module_name = getattr(module, '__name__', '')
        # Skip any torch internal modules that cause issues
        if 'torch' in module_name and any(x in module_name for x in ['_classes', 'classes', '_C']):
            return []
        try:
            return _original_get_module_paths(module)
        except Exception:
            # If any error occurs, just return empty list
            return []
    
    # Apply the patch
    streamlit.watcher.local_sources_watcher.get_module_paths = patched_get_module_paths
except Exception:
    # If patching fails, just continue
    pass

# Also suppress warnings as backup
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")






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

# Add custom CSS (consolidated)
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
        background-color: var(--background-color, #F8FAFC);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color, #E2E8F0);
        color: var(--text-color, #1a202c);
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
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
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
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark), [data-theme="dark"] {
        .section { background-color: #2D3748 !important; border-color: #4A5568 !important; color: #E2E8F0 !important; }
        .main-header { color: #63B3ED !important; }
        .sub-header { color: #90CDF4 !important; }
        .info-box { background-color: #1A365D !important; border-left-color: #63B3ED !important; color: #E2E8F0 !important; }
        .metric-card { background-color: #2D3748 !important; color: #E2E8F0 !important; }
        .metric-value { color: #90CDF4 !important; }
        .metric-label { color: #A0AEC0 !important; }
    }
</style>
""", unsafe_allow_html=True)

def show_variability_warning():
    """Display standardised warning about result variability."""
    st.warning("""
    ⚠️ **RESULT VARIABILITY NOTICE**
    
    Neural network training is inherently stochastic. Results will differ between runs due to random weight initialisation.
    Performance typically ranges 35-40% accuracy (better than random 33%). Run multiple experiments for robust assessment.
    """)

def main():
    """Main function for the Streamlit app."""
    st.markdown('<div class="main-header">Intelligent Algorithmic Trading System - ARI5123</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        An intelligent algorithmic trading system has been implemented using Convolutional Neural Networks (CNNs). 
        The approach is based upon Sezer & Ozbayoglu (2018), with modern enhancements having been incorporated.
    </div>
    """, unsafe_allow_html=True)
    
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
    The system analyses historical stock data, identifies patterns, and makes trading decisions.
    
    ### 🧪 Experimental Framework
    
    **CNN-BI Implementation Based on Sezer & Ozbayoglu (2018):**
    - Benchmark Paper Target: 7.20% annualised return (2007-2012 period)
    - Multiple Architectures: Simple and Advanced CNN configurations available
    - Signal Processing: Fixed and Dynamic threshold methods implemented
    - Test Periods: 2007-2012 (crisis) and 2012-2017 (bull market) evaluation
    
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
    
    fig = create_improved_architecture_diagram()
    st.pyplot(fig)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_data_preparation():
    """Display data preparation page with benchmark data download and visualisation."""
    st.markdown('<div class="sub-header">Data Preparation</div>', unsafe_allow_html=True)
    
    # Check if EODHD is configured
    try:
        from utils.eodhdapi import get_client
        client = get_client()
        st.success("✅ EODHD API configured successfully")
    except Exception as e:
        st.error(f"❌ EODHD API Error: {str(e)}")
        st.info("To use this application, you need an EODHD API key from https://eodhd.com")
        st.stop()
        return
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Download Benchmark Data")
    
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
            st.info("**Period 1: 2007-2012** - Includes 2008 financial crisis")
        else:
            st.info("**Period 2: 2012-2017** - Bull market period")
    
    # Download and prepare buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("1. Download Benchmark Data"):
            # Clear downstream results
            for key in ['model', 'history', 'predictions', 'strategy_results', 'benchmark_results']:
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
            
            
            #overall_progress = st.progress(0)
            #status_text = st.empty()
            
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
                        
                        st.success(f"""
                        ✅ Benchmark features prepared for {selected_ticker} ({test_period})!
                        
                        **Training Data:**
                        - Period: {splits['train_dates'][0]} to {splits['train_dates'][1]}
                        - Samples: {len(prepared_data['X_train'])} training, {len(prepared_data['X_val'])} validation
                        
                        **Testing Data:**
                        - Period: {splits['test_dates'][0]} to {splits['test_dates'][1]}
                        - Samples: {len(prepared_data['X_test'])} testing samples
                        
                        **Label Distribution:**
                        - Hold (0): {np.sum(prepared_data['y_train'] == 0)} samples
                        - Buy (1): {np.sum(prepared_data['y_train'] == 1)} samples
                        - Sell (2): {np.sum(prepared_data['y_train'] == 2)} samples
                        
                        Ready for CNN training using benchmark methodology!
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
        
        st.success(f"""
        **Benchmark preparation complete for {prepared['ticker']}!**
        
        - Test Period: {prepared.get('test_period', 'Unknown')}
        - Training Samples: {len(prepared['X_train'])}
        - Validation Samples: {len(prepared['X_val'])}
        - Testing Samples: {len(prepared['X_test'])}
        - Image Size: {prepared['X_train'].shape[1]}x{prepared['X_train'].shape[2]} pixels
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
        if X_val is not None:
            X_val = X_val.reshape(X_val.shape[0], 1, 30, 30)
        X_test = X_test.reshape(X_test.shape[0], 1, 30, 30)
        
        prepared_data['X_train'] = X_train
        prepared_data['X_val'] = X_val
        prepared_data['X_test'] = X_test
        st.session_state.prepared_data = prepared_data
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio(
            "Model Architecture",
            options=["Simple CNN", "Advanced CNN"],
            index=0,
            help="Simple CNN: Optimised for stability. Advanced CNN: More complex architecture."
        )
        
        epochs = st.slider("Training Epochs", min_value=10, max_value=200, value=100, step=5)
        
        st.markdown("#### Regularisation Settings")
        weight_decay = st.select_slider(
            "Weight Decay",
            options=[1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
            help="Higher values prevent overfitting"
        )
        
        use_early_stopping = st.checkbox("Use Early Stopping", value=True)
        patience = st.slider("Early Stopping Patience", min_value=3, max_value=10, value=8)
    
    with col2:
        batch_size = st.select_slider(
            "Batch Size",
            options=[16, 32, 64, 128],
            value=32
        )
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0002, 0.0003, 0.0005],
            value=0.0005,
            format_func=lambda x: f"{x:.4f}"
        )
        
        use_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True)
        gradient_clipping = st.checkbox("Use Gradient Clipping", value=True)
    
    # Store configuration for batch experiments
    st.session_state.model_config = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        'epochs': epochs,
        'model_type': model_type,
        'use_scheduler': use_scheduler,
        'use_early_stopping': use_early_stopping,
        'gradient_clipping': gradient_clipping
    }
    
    show_variability_warning()
    
    if st.button("Train Model"):
        # Clear downstream results
        for key in ['strategy_results', 'benchmark_results']:
            if key in st.session_state:
                del st.session_state[key]
        
        try:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            model = CNNTradingModel(input_shape=(30, 30))
            
            if model_type == "Simple CNN":
                model.build_simple_cnn()
            else:
                model.build_advanced_cnn()
            
            def progress_callback(epoch, train_loss, train_acc, val_loss, val_acc):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                progress_text.text(f"Epoch {epoch + 1}/{epochs}")
            
            history = model.train(
                X_train, y_train, X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
                use_scheduler=use_scheduler,
                use_early_stopping=use_early_stopping,
                use_gradient_clipping=gradient_clipping,
                callback=progress_callback
            )
            
            progress_bar.progress(1.0)
            progress_text.text("Training Complete!")
            
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
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            with col2:
                if 'class_metrics' in metrics and 'Buy' in metrics['class_metrics']:
                    st.metric("Buy Precision", f"{metrics['class_metrics']['Buy']['precision']:.2%}")
            with col3:
                if 'class_metrics' in metrics and 'Sell' in metrics['class_metrics']:
                    st.metric("Sell Precision", f"{metrics['class_metrics']['Sell']['precision']:.2%}")
            with col4:
                if 'class_metrics' in metrics and 'Hold' in metrics['class_metrics']:
                    st.metric("Hold Precision", f"{metrics['class_metrics']['Hold']['precision']:.2%}")
            
            # Display evaluation plots
            if y_pred_prob.shape[1] > 1:
                fig = plot_multiclass_evaluation(y_test, y_pred_prob)
                st.pyplot(fig)
            
            model.save(f"{ticker}_model")
            st.success(f"✅ Model trained and saved! Accuracy: {metrics['accuracy']:.1%}")
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_trading_strategy():
    """Display trading strategy page."""
    st.markdown('<div class="sub-header">Trading Strategy</div>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' section.")
        return
    
    if 'predictions' not in st.session_state or st.session_state.predictions is None:
        st.warning("Model predictions are not available. Please train the model first.")
        return
    
    prepared_data = st.session_state.prepared_data
    predictions = st.session_state.predictions
    
    with st.container():
        st.markdown("### Trading Strategy Configuration")
        
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
                step=5
            ) / 100
        
        with col2:
            threshold_method = st.radio(
                "Signal Threshold Method",
                options=["Fixed", "Dynamic"],
                index=1,
                help="Dynamic adapts thresholds based on signal distribution"
            )
            
            if threshold_method == "Fixed":
                signal_threshold = st.slider(
                    "Signal Threshold",
                    min_value=0.10,
                    max_value=0.30,
                    value=0.20,
                    step=0.02
                )
            else:
                st.info("Using dynamic thresholds based on signal distribution")
                signal_threshold = None
            
            position_sizing = st.selectbox(
                "Position Sizing Method",
                options=["Fixed Percentage", "Kelly Criterion (Advanced)"],
                index=0
            )
    
    show_variability_warning()
    
    if st.button("Execute Trading Strategy"):
        # Clear previous results
        for key in ['strategy_results', 'benchmark_results']:
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
            
            # Run strategies
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
            strategy_portfolio = strategy.get_portfolio_values().copy()
            strategy_trades = strategy.get_trades().copy()

            strategy.reset()
            benchmark_results = strategy.apply_buy_and_hold(prices)
            benchmark_portfolio = strategy.get_portfolio_values().copy()

            strategy.reset()
            random_results = strategy.apply_random_strategy(prices)

            # Store results
            st.session_state.strategy_results = strategy_results
            st.session_state.benchmark_results = benchmark_results
            st.session_state.random_results = random_results
            st.session_state.strategy_portfolio = strategy_portfolio
            st.session_state.benchmark_portfolio = benchmark_portfolio
            st.session_state.trades = strategy_trades
            st.session_state.test_dates = aligned_dates
            
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
                st.metric("Sharpe Ratio", f"{strategy_results['sharpe_ratio']:.2f}")
            
            with col3:
                st.metric("Max Drawdown", f"{strategy_results['max_drawdown']:.2%}")
            
            with col4:
                st.metric("Win Rate", f"{strategy_results['win_rate']:.2%}")
            
            if strategy_results['total_return'] > benchmark_results['total_return']:
                st.success(f"🎉 Strategy outperformed by {perf_delta:.2%}")
            else:
                st.warning(f"📊 Strategy underperformed by {-perf_delta:.2%}")
            
            # Portfolio performance plot
            st.markdown("#### Portfolio Performance")
            fig = plot_interactive_portfolio(
                st.session_state.strategy_portfolio,
                st.session_state.benchmark_portfolio,
                aligned_dates,
                st.session_state.trades
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

def display_performance_analysis():
    """Display performance analysis page."""
    st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
    
    if 'strategy_results' not in st.session_state or st.session_state.strategy_results is None:
        st.warning("Please execute a trading strategy first in the 'Trading Strategy' section.")
        return
    
    strategy_results = st.session_state.strategy_results
    benchmark_results = st.session_state.benchmark_results
    
    ticker = st.session_state.prepared_data['ticker']
    test_period = st.session_state.prepared_data.get('test_period', 'Unknown')
    
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

def display_batch_experiment_page():
    """Display the batch experiment page."""
    st.markdown('<div class="sub-header">Comprehensive Benchmark Experiment</div>', unsafe_allow_html=True)
    
    config = st.session_state.get('model_config', None)
    
    if config is None:
        st.warning("Please visit the 'Model Training (Single Stock)' section first to set your model parameters.")
        return
    
    st.info(f"""
    **Current Model Configuration**:
    - Architecture: {config['model_type']}
    - Epochs: {config['epochs']}
    - Learning Rate: {config['learning_rate']:.4f}
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_period = st.selectbox(
            "Select Test Period",
            options=["2007-2012", "2012-2017"]
        )
        
        quick_run = st.checkbox(
            "Quick Run (30 epochs instead of configured epochs)", 
            value=False
        )
    
    with col2:
        dow30_stocks = [
            "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
            "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
            "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
        ]
        
        # Stock code to company name mapping
        stock_names = {
            "AAPL": "Apple Inc.",
            "AXP": "American Express",
            "BA": "Boeing",
            "CAT": "Caterpillar",
            "CSCO": "Cisco Systems",
            "CVX": "Chevron",
            "DD": "DuPont",
            "DIS": "Walt Disney",
            "GE": "General Electric",
            "GS": "Goldman Sachs",
            "HD": "Home Depot",
            "IBM": "IBM",
            "INTC": "Intel",
            "JNJ": "Johnson & Johnson",
            "JPM": "JPMorgan Chase",
            "KO": "Coca-Cola",
            "MCD": "McDonald's",
            "MMM": "3M",
            "MRK": "Merck",
            "MSFT": "Microsoft",
            "NKE": "Nike",
            "PFE": "Pfizer",
            "PG": "Procter & Gamble",
            "TRV": "Travelers",
            "UNH": "UnitedHealth",
            "UTX": "United Technologies",
            "V": "Visa",
            "VZ": "Verizon",
            "WMT": "Walmart",
            "XOM": "ExxonMobil"
        }
        
        # Always use all stocks, quick_run only affects epochs
        selected_stocks = dow30_stocks
        
        if quick_run:
            st.info(f"**Testing {len(selected_stocks)} stocks with 30 epochs (Quick Run)**")
        else:
            st.info(f"**Testing {len(selected_stocks)} stocks with {config['epochs']} epochs**")
    
    # Display stock list in a permanent, always-visible section BEFORE the button
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Dow 30 Stocks to be Tested")
    
    # Create three columns for better display
    col1_stocks, col2_stocks, col3_stocks = st.columns(3)
    
    # Split stocks into three groups
    stocks_per_column = len(selected_stocks) // 3
    remainder = len(selected_stocks) % 3
    
    # Calculate split points
    split1 = stocks_per_column + (1 if remainder > 0 else 0)
    split2 = split1 + stocks_per_column + (1 if remainder > 1 else 0)
    
    # Display stocks in columns
    with col1_stocks:
        
        for ticker in selected_stocks[:split1]:
            st.markdown(f"• **{ticker}** - {stock_names[ticker]}")
    
    with col2_stocks:
        
        for ticker in selected_stocks[split1:split2]:
            st.markdown(f"• **{ticker}** - {stock_names[ticker]}")
    
    with col3_stocks:
        
        for ticker in selected_stocks[split2:]:
            st.markdown(f"• **{ticker}** - {stock_names[ticker]}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add some spacing before the button
    st.markdown("---")
    
    if st.button("🚀 Run Comprehensive Benchmark Experiment"):
        with st.spinner("Running comprehensive benchmark experiment..."):
            summary = run_batch_experiment(
                test_period=test_period,
                selected_stocks=selected_stocks,
                config=config,
                quick_run=quick_run
            )
            
            if summary:
                st.success("✅ Experiment completed successfully!")
                display_batch_results(summary, test_period)

def check_data_sufficiency(splits, ticker, window_size=30, prediction_horizon=5):
    """Check if ticker has sufficient data for analysis."""
    train_data = splits['train_data']
    test_data = splits['test_data']
    
    min_train_samples = window_size + prediction_horizon + 100
    min_test_samples = prediction_horizon + 20
    
    issues = []
    
    if len(train_data) < min_train_samples:
        issues.append(f"Insufficient training data: {len(train_data)} rows")
    
    if len(test_data) < min_test_samples:
        issues.append(f"Insufficient test data: {len(test_data)} rows")
    
    return issues

def run_batch_experiment(test_period, selected_stocks, config, quick_run=False):
    """Run batch experiment across multiple stocks."""
    from utils.data_loader import DataLoader
    from utils.feature_engineering import FeatureEngineer
    from utils.model import CNNTradingModel
    from utils.trading_strategy import TradingStrategy
    
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    
    # Download data
    with st.spinner("Downloading benchmark data..."):
        data = data_loader.download_benchmark_data(force_download=False, batch_size=10)
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(selected_stocks):
        status_text.text(f"Processing {ticker} ({i+1}/{len(selected_stocks)})")
        
        try:
            if ticker not in data:
                st.warning(f"⚠️ Skipping {ticker} - no data available")
                continue
            
            # Get data splits
            splits = data_loader.get_benchmark_splits(data, ticker, test_period)
            
            # Check data sufficiency
            data_issues = check_data_sufficiency(splits, ticker)
            if data_issues:
                st.warning(f"⚠️ Skipping {ticker}: {', '.join(data_issues)}")
                continue
            
            # Prepare features
            prepared_data = feature_engineer.prepare_benchmark_features(
                splits['train_data'], splits['test_data'], ticker
            )
            
            # Create and train model
            model = CNNTradingModel()
            
            if config['model_type'] == 'Simple CNN':
                model.build_simple_cnn()
            else:
                model.build_advanced_cnn()
            
            epochs = 30 if quick_run else config['epochs']
            
            history = model.train(
                prepared_data['X_train'], prepared_data['y_train'],
                prepared_data['X_val'], prepared_data['y_val'],
                epochs=epochs,
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate']
            )
            
            # Evaluate model
            y_pred_prob = model.predict(prepared_data['X_test'])
            metrics = model.evaluate(prepared_data['X_test'], prepared_data['y_test'])
            
            # Trading strategy
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
                test_period=test_period
            )
            
            strategy.reset()
            benchmark_results = strategy.apply_buy_and_hold(prices)
            
            outperformed = strategy_results['total_return'] > benchmark_results['total_return']
            
            result = {
                'ticker': ticker,
                'accuracy': metrics['accuracy'],
                'strategy_return': strategy_results['total_return'],
                'benchmark_return': benchmark_results['total_return'],
                'outperformed': outperformed
            }
            
            results.append(result)
            
            status_icon = "✅" if outperformed else "❌"
            st.write(f"{status_icon} **{ticker}**: Accuracy={metrics['accuracy']:.1%}, "
                    f"CNN Return={strategy_results['total_return']:.1%}, "
                    f"B&H Return={benchmark_results['total_return']:.1%}")
            
        except Exception as e:
            st.error(f"❌ Error processing {ticker}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(selected_stocks))
    
    progress_bar.empty()
    status_text.empty()
    
    # Generate summary
    if results:
        summary = {
            'num_stocks': len(results),
            'avg_accuracy': np.mean([r['accuracy'] for r in results]),
            'avg_strategy_return': np.mean([r['strategy_return'] for r in results]),
            'avg_benchmark_return': np.mean([r['benchmark_return'] for r in results]),
            'success_rate': sum(r['outperformed'] for r in results) / len(results),
            'results': results
        }
        
        return summary
    
    return None

def display_batch_results(summary, test_period):
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
    
    # CSV download
    #csv = results_df.to_csv(index=False)
    #st.download_button(
    #    label="📥 Download Results CSV",
    #    data=csv,
    #    file_name=f"benchmark_results_{test_period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    #    mime='text/csv'
    #)

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
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Hold', 'Buy', 'Sell']
    num_classes = cm.shape[0]
    used_class_names = class_names[:num_classes]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 0], 
                xticklabels=used_class_names, yticklabels=used_class_names)
    
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_title('Confusion Matrix')
    
    # ROC curves
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    
    for i, class_name in enumerate(used_class_names):
        if i < y_pred_prob.shape[1]:
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_prob[:, i]
            
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                axes[0, 1].plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
            except:
                continue
    
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True)
    
    # Class distribution
    class_counts = np.bincount(y_true, minlength=len(used_class_names))
    axes[1, 0].bar(range(len(used_class_names)), class_counts[:len(used_class_names)])
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Class Distribution')
    axes[1, 0].set_xticks(range(len(used_class_names)))
    axes[1, 0].set_xticklabels(used_class_names)
    
    # Prediction distribution
    pred_counts = np.bincount(y_pred, minlength=len(used_class_names))
    axes[1, 1].bar(range(len(used_class_names)), pred_counts[:len(used_class_names)])
    axes[1, 1].set_xlabel('Predicted Class')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].set_xticks(range(len(used_class_names)))
    axes[1, 1].set_xticklabels(used_class_names)
    
    plt.tight_layout()
    return fig

# Run the app
if __name__ == "__main__":
    main()