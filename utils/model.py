"""
CNN Model Module (PyTorch Version)

This module defines the Convolutional Neural Network (CNN) model for stock trading.
The implementation is inspired by Sezer & Ozbayoglu (2018) but with modern enhancements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import joblib
import logging

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """
    Simple CNN model similar to the one in Sezer & Ozbayoglu (2018).
    """
    def __init__(self, input_shape):
        super(SimpleCNN, self).__init__()
        # input_shape should be (sequence_length, features)
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Calculate size after convolutions and pooling
        self.flatten_size = self._get_flatten_size(input_shape)
        
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        
    def _get_flatten_size(self, input_shape):
        # Calculate the size of the flattened features after convolutions
        x = torch.zeros(1, input_shape[1], input_shape[0])
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.numel()
    
    def forward(self, x):
        # Transpose input from [batch, seq_len, features] to [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x

class SelfAttention(nn.Module):
    """
    Self-attention mechanism for time series data.
    """
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))
    
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention weights
        output = torch.matmul(attention, v)
        return output

class AdvancedCNN(nn.Module):
    """
    Advanced CNN model with attention mechanisms.
    """
    def __init__(self, input_shape):
        super(AdvancedCNN, self).__init__()
        # input_shape should be (sequence_length, features)
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Calculate size after convolutions
        self.conv_output_size = self._get_conv_output_size(input_shape)
        
        # Self-attention layer (applied after reshaping back to sequence form)
        self.attention = SelfAttention(256)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
    
    def _get_conv_output_size(self, input_shape):
        # Calculate the output size after convolutions
        x = torch.zeros(1, input_shape[1], input_shape[0])
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.size()
    
    def forward(self, x):
        # Transpose input from [batch, seq_len, features] to [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        # Convolutional blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Reshape for attention [batch, seq_len, channels]
        x_reshaped = x.permute(0, 2, 1)
        
        # Apply attention
        x_attention = self.attention(x_reshaped)
        
        # Convert back to [batch, channels, seq_len]
        x = x_attention.permute(0, 2, 1)
        
        # Global average pooling
        x = self.gap(x).squeeze(-1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x

class CNNTradingModel:
    """
    CNN-based trading model with PyTorch implementation.
    """
    
    def __init__(self, input_shape, model_dir="models"):
        """
        Initialize the CNN trading model.
        
        Args:
            input_shape: Shape of input data (window_size, features)
            model_dir: Directory to save trained models
        """
        self.input_shape = input_shape
        self.model_dir = model_dir
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def build_simple_cnn(self):
        """
        Build a simple CNN model similar to the one in Sezer & Ozbayoglu (2018).
        
        Returns:
            Compiled CNN model
        """
        model = SimpleCNN(self.input_shape).to(self.device)
        self.model = model
        return model
    
    def build_advanced_cnn(self):
        """
        Build an advanced CNN model with attention mechanisms.
        
        Returns:
            Compiled advanced CNN model
        """
        model = AdvancedCNN(self.input_shape).to(self.device)
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, advanced=True, epochs=50, batch_size=32):
        """
        Train the CNN model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            advanced: Whether to use the advanced CNN model
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Build model if not already built
        if self.model is None:
            if advanced:
                self.build_advanced_cnn()
            else:
                self.build_simple_cnn()
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training loop
        history = {
            'loss': [], 
            'accuracy': [], 
            'val_loss': [], 
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        logger.info("Starting model training...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = correct_train / total_train
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    total_val += batch_y.size(0)
                    correct_val += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = correct_val / total_val
            
            # Record history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - '
                  f'val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        logger.info("Model training completed")
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Input data
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been built and trained yet")
        
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def save(self, filename):
        """
        Save the trained model.
        
        Args:
            filename: Filename to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model state_dict
        model_path = os.path.join(self.model_dir, f"{filename}.pt")
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load(self, filename, advanced=True):
        """
        Load a trained model.
        
        Args:
            filename: Filename of the saved model
            advanced: Whether to load the advanced CNN model
        """
        model_path = os.path.join(self.model_dir, f"{filename}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Initialize the model architecture
        if advanced:
            self.build_advanced_cnn()
        else:
            self.build_simple_cnn()
        
        # Load model state_dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dict with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built and trained yet")
        
        # Get model predictions
        y_pred_prob = self.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }