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
import plotly.graph_objects as go
import os
import datetime
import torch
from torchinfo import summary as torch_summary
import logging
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    plot_stock_data, plot_model_training_history, plot_model_evaluation,
    plot_portfolio_performance, plot_interactive_portfolio, plot_strategy_comparison,
    create_performance_report
)

# Initialise session state (keep this for app functionality)
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
    st.markdown('<div class="main-header">Intelligent Algorithmic Trading System - ARI5123</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        This application implements an intelligent algorithmic trading system using Convolutional Neural Networks (CNNs).
        The approach is based on the paper "Financial Trading Model with Stock Bar Chart Image Time Series with Deep 
        Convolutional Neural Networks" by Sezer & Ozbayoglu (2018) with modern enhancements.
    </div>
    """, unsafe_allow_html=True)
    
    # Create sidebar for navigation
    st.sidebar.title("Main Menu")
    
    # Show the navigation options
    page = st.sidebar.radio("Go to", ["Introduction", "Data Preparation", "Model Training", "Trading Strategy", "Performance Analysis"])
    
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
    """Display data preparation page with data download and visualisation."""
    st.markdown('<div class="sub-header">Data Preparation</div>', unsafe_allow_html=True)
    
    # Remove debug info line
    
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
    
    # Create two separate buttons for downloading and preparing
    col1, col2 = st.columns(2)
    
    with col1:
        # Download button
        if st.button("1. Download Data"):
            with st.spinner("Downloading data..."):
                # Initialise DataLoader
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
    
    with col2:
        # Only show prepare button if data is available
        prepare_disabled = True
        if 'data' in st.session_state and st.session_state.data is not None:
            if selected_ticker in st.session_state.data:
                prepare_disabled = False
        
        # Prepare button
        if st.button("2. Prepare Data for Modeling", disabled=prepare_disabled):
            if st.session_state.data is not None and selected_ticker in st.session_state.data:
                with st.spinner("Preparing data for modeling..."):
                    try:
                        # Initialise feature engineer
                        feature_engineer = FeatureEngineer()
                        
                        # Prepare features
                        df_features = feature_engineer.prepare_features_for_model(
                            st.session_state.data, 
                            selected_ticker
                        )
                        
                        # Create sequences for the model
                        window_size = 30  # Default window size from the paper
                        
                        # Display features info
                        st.info(f"Generated {len(df_features.columns)} features for {selected_ticker}")
                        
                        # Create sequences and labels
                        X, y = [], []
                        for i in range(window_size, len(df_features)):
                            X.append(df_features.iloc[i-window_size:i].values)
                            # Label: 1 if price goes up, 0 if it goes down
                            price_change = df_features.iloc[i]['Close'] > df_features.iloc[i-1]['Close']
                            y.append(1 if price_change else 0)
                        
                        X = np.array(X)
                        y = np.array(y)
                        
                        # Split data into train, validation, and test sets
                        train_ratio = 0.7
                        val_ratio = 0.15
                        
                        # Calculate split indices
                        train_size = int(len(X) * train_ratio)
                        val_size = int(len(X) * val_ratio)
                        
                        # Create train, validation, and test sets
                        X_train, y_train = X[:train_size], y[:train_size]
                        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
                        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
                        
                        # Store prepared data in session state
                        st.session_state.prepared_data = {
                            'X_train': X_train,
                            'y_train': y_train,
                            'X_val': X_val,
                            'y_val': y_val,
                            'X_test': X_test,
                            'y_test': y_test,
                            'ticker': selected_ticker,
                            'window_size': window_size
                        }
                        
                        st.success(f"""
                        Data preparation complete! 
                        - Training set: {X_train.shape[0]} samples
                        - Validation set: {X_val.shape[0]} samples
                        - Test set: {X_test.shape[0]} samples
                        
                        You can now proceed to the Model Training page.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error preparing data: {str(e)}")
                        st.exception(e)
            else:
                st.error("Please download data first.")
    
    # Show data visualisation if data is available
    if 'data' in st.session_state and st.session_state.data is not None and selected_ticker in st.session_state.data:
        st.markdown("### Data Visualisation")
        
        # Get data for the selected ticker
        ticker_data = st.session_state.data[selected_ticker]
        
        # Create visualisation
        fig = plot_stock_data(st.session_state.data, selected_ticker)
        st.pyplot(fig)
        
        # Show data sample
        st.markdown("### Data Sample")
        st.dataframe(ticker_data.head())
    
    # Show data preparation status
    if 'prepared_data' in st.session_state and st.session_state.prepared_data is not None:
        st.markdown("### ✅ Data Preparation Status")
        st.success(f"""
        Data is prepared and ready for model training!
        - Ticker: {st.session_state.prepared_data['ticker']}
        - Training samples: {st.session_state.prepared_data['X_train'].shape[0]}
        - Features per sample: {st.session_state.prepared_data['X_train'].shape[2]}
        """)
    
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
        
        Using the same data range and stock selection ensures a fair comparison between this implementation and the benchmark results.
        """)
        
    st.markdown('</div>', unsafe_allow_html=True)

def display_model_training():
    """Display model training page with model configuration and training."""
    st.markdown('<div class="sub-header">Model Training</div>', unsafe_allow_html=True)
    
    # Check if data is prepared
    if 'prepared_data' not in st.session_state or st.session_state.prepared_data is None:
        st.warning("Please prepare data first in the 'Data Preparation' section.")
        st.info("Go to the Data Preparation page, download data, and click the 'Prepare Data for Modeling' button.")
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
    
    # Ensure data is in the right shape for CNN [batch, channels, height, width]
    if len(X_train.shape) == 3 and X_train.shape[1:] == (30, 30):
        # Input is [batch, height, width], need to add channel dimension
        X_train = X_train.reshape(X_train.shape[0], 1, 30, 30)
        X_val = X_val.reshape(X_val.shape[0], 1, 30, 30)
        X_test = X_test.reshape(X_test.shape[0], 1, 30, 30)
        # Update the session state
        prepared_data['X_train'] = X_train
        prepared_data['X_val'] = X_val
        prepared_data['X_test'] = X_test
        st.session_state.prepared_data = prepared_data
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Model Configuration")
    
    # Show data summary
    st.info(f"""
    **Data Summary for {ticker}**
    - Training samples: {X_train.shape[0]} (with {X_train.shape[2] if len(X_train.shape) > 2 else 30} features)
    - Validation samples: {X_val.shape[0]}
    - Test samples: {X_test.shape[0]} 
    """)
    
    # Create progress bar components ahead of time (but initially hidden)
    progress_container = st.empty()
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio(
            "Model Architecture",
            options=["Simple CNN", "Advanced CNN with Attention"],
            index=0  # Use Simple CNN as default to match paper
        )
        
        epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=50, step=5)
        
        # Add new hyperparameters
        st.markdown("#### Advanced Settings")
        use_focal_loss = st.checkbox("Use Focal Loss (for imbalanced data)", value=True)
        use_class_weights = st.checkbox("Use Class Weights", value=True)
        weight_decay = st.select_slider(
            "Weight Decay (Regularisation)",
            options=[0, 1e-6, 1e-5, 1e-4, 1e-3],
            value=1e-5,
            format_func=lambda x: f"{x:.0e}" if x > 0 else "0"
        )
    
    with col2:
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128],
            value=32
        )
        
        use_early_stopping = st.checkbox("Use Early Stopping", value=True)
        
        # Add learning rate selection
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
            value=0.001,
            format_func=lambda x: f"{x:.4f}"
        )
    
    # Check input shape and update
    input_shape = X_train.shape[1:] if len(X_train.shape) > 1 else (30, 30)
    
    # Train model button
    if st.button("Train Model"):
        try:
            # Initialise model
            if len(input_shape) == 3 and input_shape[0] == 1:
                # Input already has channel dimension [channels, height, width]
                model = CNNTradingModel(input_shape=input_shape[1:])
            else:
                # Input is [height, width] or needs to be adjusted
                model = CNNTradingModel(input_shape=(30, 30))
            
            # Build model
            if model_type == "Simple CNN":
                model.build_simple_cnn()
                advanced = False
            else:
                model.build_advanced_cnn()
                advanced = True
            
            # Now show the progress elements before showing the model summary
            with progress_container.container():
                st.markdown("#### Training Progress")
                progress_text = st.empty()
                progress_bar = st.progress(0)
                metrics_text = st.empty()
                
                # Initialise with starting values
                progress_text.markdown("**Preparing to train: 0/{} epochs (0%)**".format(epochs))
            
            # Display model summary
            st.markdown("#### Model Summary")
            try:
                # Create a sample input tensor with the right shape for 2D CNN
                if model_type == "Simple CNN":
                    # For SimpleCNN which expects [batch, channels, height, width]
                    sample_input = torch.zeros((1, 1, 30, 30))
                else:
                    # For AdvancedCNN
                    sample_input = torch.zeros((1, 1, 30, 30))
                    
                summary_str = str(torch_summary(model.model, input_size=sample_input.shape))
                st.code(summary_str)
            except Exception as e:
                st.warning(f"Could not display detailed model summary due to: {str(e)}")
                
                # Provide a manual summary instead
                if model_type == "Simple CNN":
                    st.code("""
                    SimpleCNN(
                      (conv1): Conv2d(1, 32, kernel_size=3, stride=1)
                      (pool1): MaxPool2d(kernel_size=2, stride=2)
                      (dropout1): Dropout(p=0.2)
                      (conv2): Conv2d(32, 64, kernel_size=3, stride=1)
                      (pool2): MaxPool2d(kernel_size=2, stride=2)
                      (dropout2): Dropout(p=0.2)
                      (fc1): Linear(in_features=64*6*6, out_features=128)
                      (dropout3): Dropout(p=0.3)
                      (fc2): Linear(in_features=128, out_features=3)
                    )
                    """)
                else:
                    st.code("""
                    AdvancedCNN(
                      (conv1): Conv2d(1, 32, kernel_size=3, padding=1)
                      (bn1): BatchNorm2d(32)
                      (pool1): MaxPool2d(kernel_size=2)
                      (res_conv1a): Conv2d(32, 64, kernel_size=3, padding=1)
                      (res_bn1a): BatchNorm2d(64)
                      ...
                      (fc3): Linear(in_features=64, out_features=3)
                    )
                    """)
            
            # Custom callback for updating progress
            class ProgressCallback:
                def __init__(self, total_epochs):
                    self.total_epochs = total_epochs
                    self.current_epoch = 0
                
                def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
                    self.current_epoch = epoch
                    # Update progress bar (convert to 0-1 range)
                    progress = min(1.0, (epoch + 1) / self.total_epochs)
                    progress_bar.progress(progress)
                    # Update text
                    progress_text.markdown(f"**Training Progress: {epoch+1}/{self.total_epochs} epochs completed ({progress*100:.1f}%)**")
                    metrics_text.markdown(f"""
                        **Current metrics:**  
                        - Training loss: {train_loss:.4f}
                        - Training accuracy: {train_acc:.2%}
                        - Validation loss: {val_loss:.4f}
                        - Validation accuracy: {val_acc:.2%}
                    """)
            
            # Create callback instance
            callback = ProgressCallback(epochs)
            
            # Patch the model's train method to use our progress callback
            original_train = model.train
            
            def train_with_progress(*args, **kwargs):
                """
                Train the CNN model with progress updates.
                """
                # Initialise progress
                progress_text.markdown("**Training Progress: 0/{} epochs completed (0.0%)**".format(epochs))
                progress_bar.progress(0)
                
                # First, let's rebuild our model with the correct architecture
                if model_type == "Simple CNN":
                    # Create a fixed version of SimpleCNN that properly handles channel dimensions
                    class FixedSimpleCNN(nn.Module):
                        def __init__(self):
                            super(FixedSimpleCNN, self).__init__()
                            # Input is 1 channel image
                            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
                            self.pool1 = nn.MaxPool2d(kernel_size=2)
                            self.dropout1 = nn.Dropout(0.2)
                            
                            # Second conv layer takes 32 input channels (output from first conv)
                            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
                            self.pool2 = nn.MaxPool2d(kernel_size=2)
                            self.dropout2 = nn.Dropout(0.2)
                            
                            # We'll define the fc1 layer in the forward pass
                            self.fc1 = None
                            self.dropout3 = nn.Dropout(0.3)
                            self.fc2 = nn.Linear(128, 3)  # 3 classes: Buy, Hold, Sell
                            
                        def forward(self, x):
                            # Ensure input is in the right shape [batch, channels, height, width]
                            if len(x.shape) == 3:  # [batch, height, width]
                                x = x.unsqueeze(1)  # Add channel dimension
                            elif len(x.shape) == 2:  # [height, width]
                                x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                            
                            # First convolutional block
                            x = F.relu(self.conv1(x))
                            x = self.pool1(x)
                            x = self.dropout1(x)
                            
                            # Second convolutional block
                            x = F.relu(self.conv2(x))
                            x = self.pool2(x)
                            x = self.dropout2(x)
                            
                            # Flatten
                            x_flat = x.view(x.size(0), -1)
                            
                            # Create fc1 layer if it doesn't exist yet
                            if self.fc1 is None:
                                self.fc1 = nn.Linear(x_flat.shape[1], 128).to(x.device)
                            
                            # Fully connected layers
                            x = F.relu(self.fc1(x_flat))
                            x = self.dropout3(x)
                            x = self.fc2(x)
                            
                            return F.softmax(x, dim=1)  # Output probabilities for 3 classes
                            
                    # Replace the model with our fixed version
                    model.model = FixedSimpleCNN().to(model.device)
                else:
                    # The advanced CNN model should already handle channels correctly
                    pass
                
                # Training and validation datasets
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                
                # Check if labels are one-hot encoded or just indices
                if len(y_train.shape) == 1 or y_train.shape[1] == 1:
                    # Convert to integers and then to tensor
                    y_train_tensor = torch.tensor(y_train.astype(int), dtype=torch.long)
                    if y_train_tensor.dim() > 1:  # If shape is [N, 1]
                        y_train_tensor = y_train_tensor.squeeze(1)
                else:
                    # One-hot encoded, keep as is
                    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                
                # Do the same for validation data
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                if len(y_val.shape) == 1 or y_val.shape[1] == 1:
                    y_val_tensor = torch.tensor(y_val.astype(int), dtype=torch.long)
                    if y_val_tensor.dim() > 1:  # If shape is [N, 1]
                        y_val_tensor = y_val_tensor.squeeze(1)
                else:
                    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
                
                # Move tensors to device
                X_train_tensor = X_train_tensor.to(model.device)
                y_train_tensor = y_train_tensor.to(model.device)
                X_val_tensor = X_val_tensor.to(model.device)
                y_val_tensor = y_val_tensor.to(model.device)
                
                # Ensure tensors have the right shape for CNN [batch, channel, height, width]
                if len(X_train_tensor.shape) == 3:  # [batch, height, width]
                    X_train_tensor = X_train_tensor.unsqueeze(1)  # Add channel dimension
                if len(X_val_tensor.shape) == 3:  # [batch, height, width]
                    X_val_tensor = X_val_tensor.unsqueeze(1)  # Add channel dimension
                    
                # Create datasets and dataloaders
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
                
                # Regular dataloaders
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
                
                # Define optimiser with weight decay for regularisation
                optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                
                # Define learning rate scheduler
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=5, 
                   # verbose=True
                )
                
                # Define loss function for multi-class classification
                if use_focal_loss:
                    # Focal loss for multi-class classification
                    class FocalLoss(torch.nn.Module):
                        def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
                            super(FocalLoss, self).__init__()
                            self.alpha = alpha
                            self.gamma = gamma
                            self.reduction = reduction
                            # Use CrossEntropyLoss for multi-class classification
                            self.ce = torch.nn.CrossEntropyLoss(reduction='none')
                            
                        def forward(self, inputs, targets):
                            # If targets are one-hot, convert to indices
                            if targets.dim() > 1 and targets.shape[1] > 1:
                                targets = torch.argmax(targets, dim=1)
                            
                            # Calculate loss
                            CE_loss = self.ce(inputs, targets)
                            pt = torch.exp(-CE_loss)
                            F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
                            
                            if self.reduction == 'mean':
                                return torch.mean(F_loss)
                            elif self.reduction == 'sum':
                                return torch.sum(F_loss)
                            else:
                                return F_loss
                    
                    criterion = FocalLoss(alpha=0.25, gamma=2.0)
                else:
                    # Plain CrossEntropyLoss for multi-class classification
                    criterion = torch.nn.CrossEntropyLoss()
                
                # Training loop
                history = {
                    'loss': [], 
                    'accuracy': [], 
                    'val_loss': [], 
                    'val_accuracy': [],
                    'lr': []
                }
                
                best_val_loss = float('inf')
                patience = 10 if use_early_stopping else epochs
                patience_counter = 0
                best_model_state = None
                
                for epoch in range(epochs):
                    # Training phase
                    model.model.train()
                    train_loss = 0
                    correct_train = 0
                    total_train = 0
                    
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = model.model(batch_X)
                        
                        # Handle targets based on output format
                        if outputs.shape[1] == 3:  # Multi-class classification
                            # Convert one-hot to indices if needed
                            if batch_y.dim() > 1 and batch_y.shape[1] > 1:
                                batch_y = torch.argmax(batch_y, dim=1)
                        
                        # Calculate loss and backpropagate
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        
                        # Get predictions based on output format
                        if outputs.shape[1] == 3:  # Multi-class
                            predicted = torch.argmax(outputs, dim=1)
                            if batch_y.dim() > 1 and batch_y.shape[1] > 1:
                                targets = torch.argmax(batch_y, dim=1)
                            else:
                                targets = batch_y
                        else:  # Binary
                            predicted = (outputs > 0.5).float()
                            targets = batch_y
                            
                        total_train += targets.size(0)
                        correct_train += (predicted == targets).sum().item()
                    
                    train_loss /= len(train_loader)
                    train_accuracy = correct_train / total_train
                    
                    # Validation phase
                    model.model.eval()
                    val_loss = 0
                    correct_val = 0
                    total_val = 0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            # Forward pass
                            outputs = model.model(batch_X)
                            
                            # Handle targets based on output format
                            if outputs.shape[1] == 3:  # Multi-class classification
                                # Convert one-hot to indices if needed
                                if batch_y.dim() > 1 and batch_y.shape[1] > 1:
                                    batch_y = torch.argmax(batch_y, dim=1)
                            
                            loss = criterion(outputs, batch_y)
                            
                            val_loss += loss.item()
                            
                            # Get predictions based on output format
                            if outputs.shape[1] == 3:  # Multi-class
                                predicted = torch.argmax(outputs, dim=1)
                                if batch_y.dim() > 1 and batch_y.shape[1] > 1:
                                    targets = torch.argmax(batch_y, dim=1)
                                else:
                                    targets = batch_y
                            else:  # Binary
                                predicted = (outputs > 0.5).float()
                                targets = batch_y
                                
                            total_val += targets.size(0)
                            correct_val += (predicted == targets).sum().item()
                    
                    val_loss /= len(val_loader)
                    val_accuracy = correct_val / total_val
                    
                    # Record history
                    history['loss'].append(train_loss)
                    history['accuracy'].append(train_accuracy)
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_accuracy)
                    history['lr'].append(optimizer.param_groups[0]['lr'])
                    
                    # Update progress
                    callback.update(epoch, train_loss, train_accuracy, val_loss, val_accuracy)
                    
                    # Update learning rate
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience and use_early_stopping:
                            metrics_text.markdown(f"**Early stopping triggered after {epoch+1} epochs**")
                            break
                
                # Restore best model
                if best_model_state is not None:
                    model.model.load_state_dict(best_model_state)
                
                # Set final progress
                progress_bar.progress(1.0)
                if use_early_stopping and patience_counter >= patience:
                    progress_text.markdown(f"**Training Complete: Stopped early at {epoch+1}/{epochs} epochs**")
                else:
                    progress_text.markdown(f"**Training Complete: {epochs}/{epochs} epochs**")
                
                return history
            
            # Replace the train method temporarily
            model.train = train_with_progress
            
            # Train model with progress updates and new parameters
            history = model.train(
                X_train, y_train, X_val, y_val,
                advanced=advanced,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Restore original train method
            model.train = original_train
            
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
                st.metric("Buy Precision", f"{metrics['class_metrics']['Buy']['precision']:.2%}" if 'class_metrics' in metrics else "N/A")
            with col3:
                st.metric("Sell Precision", f"{metrics['class_metrics']['Sell']['precision']:.2%}" if 'class_metrics' in metrics else "N/A")
            with col4:
                st.metric("Hold Precision", f"{metrics['class_metrics']['Hold']['precision']:.2%}" if 'class_metrics' in metrics else "N/A")
            
            # Display evaluation plots
            if y_pred_prob.shape[1] > 1:  # If we have multiple classes
                fig = plot_multiclass_evaluation(y_test, y_pred_prob)
            else:  # Binary classification
                fig = plot_model_evaluation(y_test, y_pred_prob)
            st.pyplot(fig)
            
            # Save model
            model.save(f"{ticker}_model")
            st.success(f"Model trained and saved as {ticker}_model.h5")
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            st.exception(e)
    
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
        
        # Add threshold controls here
        threshold_method = st.radio(
            "Signal Threshold Method",
            options=["Fixed", "Dynamic"],
            index=1
        )
        
        signal_threshold = 0.5  # Default value
        if threshold_method == "Fixed":
            signal_threshold = st.slider(
                "Signal Threshold",
                min_value=0.3,
                max_value=0.7,
                value=0.5,
                step=0.05
            )
        else:
            st.info("Using dynamic thresholds based on signal volatility")
    
    # Execute strategy button
    if st.button("Execute Trading Strategy"):
        with st.spinner("Executing trading strategy..."):
            # Initialise trading strategy
            strategy = TradingStrategy(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost
            )
            
            # Get test set prices
            test_prices = df.loc[test_dates]['Close'].values
            
            # Create market data for enhanced strategy (if needed)
            market_data = None
            if hasattr(df, 'ATR_14'):
                market_data = df.loc[test_dates]
            
            # For multi-class predictions, we need to convert them to a signal format
            # Buy = 1, Hold = 0, Sell = 2
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Get the most probable class for each prediction
                signals = np.argmax(predictions, axis=1)
            else:
                # For binary predictions, flatten
                signals = predictions.flatten()
            
            # Ensure signals and test_prices have the same length
            min_length = min(len(signals), len(test_prices))
            signals = signals[:min_length]
            test_prices = test_prices[:min_length]
            aligned_dates = test_dates[:min_length]
            
            # Apply trading strategy with threshold option
            strategy_results = strategy.apply_strategy(
                test_prices, 
                signals, 
                aligned_dates,
                fixed_threshold=None if threshold_method == "Dynamic" else signal_threshold
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
    
    # Create metrics display directly
    metrics = [
        ('Total Return', f"{strategy_results['total_return']:.2%}", 
         f"{benchmark_results['total_return']:.2%}" if benchmark_results else None),
        ('Annualised Return', f"{strategy_results['annualized_return']:.2%}", 
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
        2. **Model Optimisation**: Fine-tune model hyperparameters and architecture
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
        4. **Transaction Cost Optimisation**: Reduce trading frequency to minimise costs
        5. **Alternative Approaches**: Consider other AI/ML techniques that may be more suitable
        """)
    st.markdown('</div>', unsafe_allow_html=True)

def plot_multiclass_evaluation(y_true, y_pred_prob, figsize=(16, 12)):
    """
    Plot model evaluation metrics for multi-class classification.
    
    Args:
        y_true: True labels (integer class labels)
        y_pred_prob: Predicted probabilities (array of shape [n_samples, n_classes])
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert one-hot encoded y_true to integer class labels if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Get predicted class
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Map the class indices to class names for clarity
    class_names = ['Hold', 'Buy', 'Sell']
    num_classes = cm.shape[0]  # Get actual number of classes from confusion matrix
    
    # Only use as many class names as we have classes in the confusion matrix
    used_class_names = class_names[:num_classes]
    
    # Let seaborn handle the tick labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 0], 
                xticklabels=used_class_names, yticklabels=used_class_names)
    
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_title('Confusion Matrix')
    
    # Plot ROC curves for each class (one-vs-rest)
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    
    # Calculate ROC curve and AUC for each class
    for i, class_name in enumerate(used_class_names):
        if i < y_pred_prob.shape[1]:  # Make sure we have predictions for this class
            # For each class, treat it as a binary classification problem
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_prob[:, i]
            
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                axes[0, 1].plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                # Skip this class if there's an error
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
    
    # Add count labels on top of bars
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
    
    # Add count labels on top of bars
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