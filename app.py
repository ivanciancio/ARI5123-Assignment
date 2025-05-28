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

# Set page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Intelligent Algorithmic Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import custom modules
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

def main():
    """Main function for the Streamlit app."""
    st.markdown('<div class="main-header">Intelligent Algorithmic Trading System - ARI5123</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        This application implements an intelligent algorithmic trading system using Convolutional Neural Networks (CNNs).
        The approach is based on Sezer & Ozbayoglu (2018) with modern enhancements.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Workflow:**
    1. **Data Preparation → Model Training → Trading Strategy → Performance Analysis:** Detailed single-stock analysis
    2. **Comprehensive Benchmark Experiment:** Multi-stock comparison with benchmark paper
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
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Download Benchmark Data")
    
    st.markdown("""
    <div class="info-box">
        For accurate benchmarking against Sezer & Ozbayoglu (2018), we use the same data range and methodology:
        - Full dataset: January 1997 to December 2017
        - Test Period 1: 2007-2012 (includes financial crisis)
        - Test Period 2: 2012-2017 (bull market period)
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
            # Create a container for detailed progress
            progress_container = st.container()
            
            with progress_container:
                st.markdown("### 📊 Download Progress")
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
                    rate_limit_info = st.empty()
                    rate_limit_info.info("🕐 Rate limit delay: Variable")
                
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
                        rate_limit_info.warning(f"🕐 {status}")
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
                    
                    # Update rate limit info if provided
                    if extra_info and "delay" in extra_info:
                        rate_limit_info.info(f"🕐 Next delay: {extra_info['delay']:.1f}s")
                
                # Initialize data loader and download
                data_loader = DataLoader()
                
                try:
                    with st.spinner("Initializing download..."):
                        data = data_loader.download_benchmark_data(
                            force_download=force_download,
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
                            - Total time: Check logs for details
                            """)
                            
                            if selected_ticker in data:
                                st.info(f"""
                                **{selected_ticker} Data Details:**
                                - Start Date: {data[selected_ticker].index[0].strftime('%Y-%m-%d')}
                                - End Date: {data[selected_ticker].index[-1].strftime('%Y-%m-%d')}
                                - Total Trading Days: {len(data[selected_ticker])}
                                """)
                        else:
                            st.error("❌ No data was successfully downloaded. Please try again later.")
                            
                except Exception as e:
                    st.error(f"❌ Error during download: {str(e)}")
                    st.exception(e)
    
    with col2:
        prepare_disabled = True
        if 'data' in st.session_state and st.session_state.data is not None:
            if selected_ticker in st.session_state.data:
                prepare_disabled = False
        
        if st.button("2. Prepare Benchmark Features", disabled=prepare_disabled):
            if st.session_state.data is not None and selected_ticker in st.session_state.data:
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
        
        ax1.set_title(f"{selected_ticker} Stock Price - Benchmark Periods")
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
        X_val = X_val.reshape(X_val.shape[0], 1, 30, 30)
        X_test = X_test.reshape(X_test.shape[0], 1, 30, 30)
        prepared_data['X_train'] = X_train
        prepared_data['X_val'] = X_val
        prepared_data['X_test'] = X_test
        st.session_state.prepared_data = prepared_data
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Model Configuration")
    
    st.info(f"""
    **Data Summary for {ticker}**
    - Training samples: {X_train.shape[0]}
    - Validation samples: {X_val.shape[0]}
    - Test samples: {X_test.shape[0]}
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio(
            "Model Architecture",
            options=["Simple CNN (Recommended)", "Advanced CNN"],
            index=0,
            help="Simple CNN prevents overfitting on small datasets. Advanced CNN has more parameters."
        )
        
        epochs = st.slider("Training Epochs", min_value=10, max_value=200, value=100, step=5)
        
        st.markdown("#### Regularization Settings")
        weight_decay = st.select_slider(
            "Weight Decay",
            options=[1e-5, 1e-4, 1e-3, 1e-2],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}",
            help="Higher values prevent overfitting (1e-4 recommended)"
        )
        
        use_early_stopping = st.checkbox("Use Early Stopping", value=True)
        patience = st.slider("Early Stopping Patience", min_value=3, max_value=15, value=5, 
                           help="Lower values stop training sooner to prevent overfitting")
    
    with col2:
        batch_size = st.select_slider(
            "Batch Size",
            options=[16, 32, 64, 128],
            value=32
        )
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0003, 0.001, 0.003],
            value=0.001,
            format_func=lambda x: f"{x:.4f}"
        )
        
        use_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True, 
                                  help="Reduces learning rate when validation loss plateaus")
        
        gradient_clipping = st.checkbox("Use Gradient Clipping", value=True,
                                      help="Prevents exploding gradients")
    
    st.warning("""
    ⚠️ **Overfitting Prevention Enabled**
    
    The model includes several anti-overfitting measures:
    - Reduced model complexity (fewer parameters)
    - Increased dropout rates (0.5, 0.7)
    - Batch normalisation
    - Early stopping on validation loss
    - Learning rate scheduling
    
    **Expected results**: Lower training accuracy (~60-70%) but better validation accuracy (~45-55%)
    """)
    
    if st.button("Train Model"):
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
                - Batch Normalization: Yes
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
                🟢 **Good Generalization**
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
    """Display trading strategy page with optimised defaults."""
    st.markdown('<div class="sub-header">Trading Strategy</div>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' section.")
        return
    
    if 'predictions' not in st.session_state or st.session_state.predictions is None:
        st.warning("Model predictions are not available. Please train the model first.")
        return
    
    prepared_data = st.session_state.prepared_data
    predictions = st.session_state.predictions
    
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
            options=["Fixed (Recommended)", "Dynamic"],
            index=0,
            help="Fixed thresholds performed better in your testing"
        )
        
        if threshold_method == "Fixed (Recommended)":
            signal_threshold = st.slider(
                "Signal Threshold",
                min_value=0.10,
                max_value=0.30,
                value=0.15,
                step=0.02,
                help="Lower values = more trades, higher values = fewer trades. Your model needs lower thresholds due to low confidence."
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
    
    # Show prediction analysis
    st.markdown("#### 🔍 Prediction Analysis")
    if len(predictions.shape) > 1:
        max_probs = np.max(predictions, axis=1)
        st.write(f"**Prediction Confidence:**")
        st.write(f"- Average: {np.mean(max_probs):.1%}")
        st.write(f"- Range: {np.min(max_probs):.1%} to {np.max(max_probs):.1%}")
        st.write(f"- Above 80%: {np.sum(max_probs > 0.8) / len(max_probs):.1%}")
        st.write(f"- Above 50%: {np.sum(max_probs > 0.5) / len(max_probs):.1%}")
    
    if st.button("Execute Trading Strategy"):
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
            
            strategy = TradingStrategy(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                max_position_size=max_position_size
            )
            
            strategy_results = strategy.apply_strategy(
                prices, signals, aligned_dates,
                fixed_threshold=signal_threshold
            )
            
            strategy.reset()
            benchmark_results = strategy.apply_buy_and_hold(prices)
            
            strategy.reset()
            random_results = strategy.apply_random_strategy(prices)
            
            # Run CNN strategy
            strategy_results = strategy.apply_strategy(prices, signals, aligned_dates)
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

            # Store all results in session state
            st.session_state.strategy_results = strategy_results
            st.session_state.benchmark_results = benchmark_results
            st.session_state.random_results = random_results
            st.session_state.strategy_portfolio = strategy_portfolio  # CNN strategy portfolio
            st.session_state.benchmark_portfolio = benchmark_portfolio  # Buy-and-hold portfolio
            st.session_state.trades = strategy_trades  # CNN strategy trades
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
                - Add more regularization
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
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_performance_analysis():
    """Display performance analysis page with corrected metrics."""
    st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
    
    if 'strategy_results' not in st.session_state or st.session_state.strategy_results is None:
        st.warning("Please execute a trading strategy first in the 'Trading Strategy' section.")
        return
    
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
        paper_success_rate = "76.7%"
    else:
        paper_accuracy = "53.8%"
        paper_cnn_return = "5.84%"
        paper_bah_return = "13.25%"
        paper_cnn_drawdown = "19.5%"
        paper_bah_drawdown = "6.97%"
        paper_success_rate = "44.8%"
    
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
        'Your Results': [
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
            
            **Key Issues Identified:**
            - Model accuracy around 30-35% (close to random for 3-class problem)
            - High validation loss suggesting overfitting
            - Excessive trading (60+ trades) leading to transaction costs
            - Poor timing of buy/sell signals
            """)
        else:
            st.success(f"""
            ✅ **Strategy Outperformed**: The CNN strategy outperformed buy-and-hold by 
            **{outperformance:.2%}** during the {test_period} period.
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_batch_experiment_page():
    """Display the batch experiment page for comprehensive benchmarking."""
    st.markdown('<div class="sub-header">Comprehensive Benchmark Experiment</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Run comprehensive benchmarking experiments across multiple stocks to generate results 
        comparable to Sezer & Ozbayoglu (2018). This provides statistical significance for 
        academic comparison and validation of your approach.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_period = st.selectbox(
            "Select Test Period",
            options=["2007-2012", "2012-2017"],
            help="Choose the benchmark test period"
        )
        
        max_stocks = st.slider(
            "Number of Stocks to Test",
            min_value=10,
            max_value=25,
            value=20,
            help="More stocks provide better statistical significance (paper used 29 stocks)"
        )
    
    with col2:
        dow30_stocks = [
            "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
            "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
            "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
        ]
        
        stock_selection = st.multiselect(
            "Select Specific Stocks (optional)",
            options=dow30_stocks,
            default=dow30_stocks[:20],
            help="Leave empty to use default selection. Paper used all Dow 30 stocks."
        )
        
        quick_run = st.checkbox(
            "Quick Run (Reduced Epochs)", 
            value=False,
            help="Use fewer epochs for faster testing (30 vs 100 epochs)"
        )
    
    # Show paper comparison info
    st.markdown("### 📋 Benchmark Paper Comparison")
    if test_period == "2007-2012":
        st.info("""
        **Sezer & Ozbayoglu (2018) Results for 2007-2012:**
        - Average Classification Accuracy: 55.2%
        - CNN Strategy Return: 7.20%
        - Buy & Hold Return: 5.86%
        - Success Rate: 76.7% (23/30 stocks outperformed)
        - Stocks Tested: 29 Dow 30 stocks
        """)
    else:
        st.info("""
        **Sezer & Ozbayoglu (2018) Results for 2012-2017:**
        - Average Classification Accuracy: 53.8%
        - CNN Strategy Return: 5.84%
        - Buy & Hold Return: 13.25%
        - Success Rate: 44.8% (13/29 stocks outperformed)
        - Stocks Tested: 29 Dow 30 stocks
        """)
    
    if st.button("🚀 Run Comprehensive Benchmark Experiment"):
        if max_stocks < 10:
            st.warning("⚠️ Consider testing at least 10 stocks for meaningful statistical comparison with the benchmark paper.")
        
        selected_stocks = stock_selection if stock_selection else None
        
        with st.spinner("Running comprehensive benchmark experiment..."):
            summary = run_batch_experiment_direct(
                test_period=test_period,
                selected_stocks=selected_stocks,
                max_stocks=max_stocks,
                quick_run=quick_run
            )
            
            if summary:
                st.session_state.batch_results = summary
                st.success("✅ Comprehensive benchmark experiment completed successfully!")
                
                display_comprehensive_benchmark_comparison(summary, test_period)
    
    if 'batch_results' in st.session_state:
        st.markdown("### 📊 Previous Comprehensive Results")
        summary = st.session_state.batch_results
        
        st.info(f"""
        **Last Experiment Summary:**
        - Test Period: {summary['test_period']}
        - Stocks Tested: {summary['num_stocks']}
        - Average Accuracy: {summary['avg_accuracy']:.1%}
        - Success Rate vs Buy & Hold: {summary['success_rate']:.1%}
        - Average Outperformance: {summary['avg_outperformance']:.2%}
        """)

def display_comprehensive_benchmark_comparison(summary, test_period):
    """Display comprehensive comparison with benchmark paper."""
    st.markdown("### 📊 Comprehensive Benchmark Analysis")
    
    benchmark_paper = get_benchmark_paper_results_fixed(test_period)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Your Results")
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

def run_batch_experiment_direct(test_period, selected_stocks, max_stocks, quick_run):
    """Run batch experiment directly without separate module."""
    try:
        from utils.data_loader import DataLoader
        from utils.feature_engineering import FeatureEngineer
        from utils.model import CNNTradingModel
        from utils.trading_strategy import TradingStrategy
        
        if selected_stocks is None:
            selected_stocks = [
                "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
                "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
                "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
            ][:max_stocks]
        
        st.info(f"Starting batch experiment for {test_period} with {len(selected_stocks)} stocks")
        
        data_loader = DataLoader()
        feature_engineer = FeatureEngineer()
        
        with st.spinner("Downloading benchmark data..."):
            data = data_loader.download_benchmark_data(force_download=False)
        
        results = {}
        successful_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(selected_stocks):
            if ticker not in data:
                st.warning(f"Skipping {ticker} - no data available")
                continue
            
            status_text.text(f"Processing {ticker} ({i+1}/{len(selected_stocks)})")
            
            try:
                splits = data_loader.get_benchmark_splits(data, ticker, test_period)
                
                prepared_data = feature_engineer.prepare_benchmark_features(
                    splits['train_data'], splits['test_data'], ticker
                )
                
                model = CNNTradingModel()
                model.build_simple_cnn()
                
                epochs = 30 if quick_run else 50
                
                def simple_callback(epoch, train_loss, train_acc, val_loss, val_acc):
                    if epoch % 10 == 0:
                        status_text.text(f"Processing {ticker} - Epoch {epoch+1}/{epochs}")
                
                history = model.train(
                    prepared_data['X_train'], prepared_data['y_train'],
                    prepared_data['X_val'], prepared_data['y_val'],
                    epochs=epochs,
                    batch_size=32,
                    learning_rate=0.001,
                    callback=simple_callback
                )
                
                y_pred_prob = model.predict(prepared_data['X_test'])
                metrics = model.evaluate(prepared_data['X_test'], prepared_data['y_test'])
                
                strategy = TradingStrategy(initial_capital=10000, transaction_cost=0.001)
                
                test_data = splits['test_data']
                test_prices = test_data['Close'].values
                test_dates = test_data.index
                
                min_len = min(len(y_pred_prob), len(test_prices))
                signals = y_pred_prob[:min_len]
                prices = test_prices[:min_len]
                dates = test_dates[:min_len]
                
                strategy_results = strategy.apply_strategy(prices, signals, dates)
                
                strategy.reset()
                benchmark_results = strategy.apply_buy_and_hold(prices)
                
                outperformed = strategy_results['total_return'] > benchmark_results['total_return']
                
                result = {
                    'ticker': ticker,
                    'classification_accuracy': metrics['accuracy'],
                    'strategy_results': strategy_results,
                    'benchmark_results': benchmark_results,
                    'model_metrics': metrics,
                    'test_period': test_period,
                    'outperformed_benchmark': outperformed
                }
                
                results[ticker] = result
                successful_results.append(result)
                
                status_icon = "✅" if outperformed else "❌"
                st.write(f"{status_icon} **{ticker}**: Accuracy={metrics['accuracy']:.1%}, "
                        f"CNN Return={strategy_results['total_return']:.1%}, "
                        f"B&H Return={benchmark_results['total_return']:.1%}, "
                        f"Outperformed: {'Yes' if outperformed else 'No'}")
                
            except Exception as e:
                st.error(f"❌ Error processing {ticker}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(selected_stocks))
        
        progress_bar.empty()
        status_text.empty()
        
        if successful_results:
            summary = generate_batch_summary_direct(successful_results, test_period)
            return summary
        else:
            st.error("No successful results generated")
            return None
            
    except Exception as e:
        st.error(f"Error in batch experiment: {str(e)}")
        st.exception(e)
        return None

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