"""
CNN Model Module (PyTorch Version)

This module defines the Convolutional Neural Network (CNN) model for stock trading.
Implements approach from Sezer & Ozbayoglu (2018) paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """
    Simple CNN model following the architecture in Sezer & Ozbayoglu (2018).
    This is a 2D CNN for bar chart images with dynamic size handling.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input is 1 channel image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
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

class AdvancedCNN(nn.Module):
    """
    Advanced CNN model with residual connections and attention mechanisms.
    This extends the base model from the paper with modern enhancements.
    """
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        
        # Input is 1 channel 30x30 image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Residual block 1
        self.res_conv1a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.res_bn1a = nn.BatchNorm2d(64)
        self.res_conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res_bn1b = nn.BatchNorm2d(64)
        self.shortcut1 = nn.Conv2d(32, 64, kernel_size=1)
        
        # Pooling after residual block
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Residual block 2
        self.res_conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res_bn2a = nn.BatchNorm2d(128)
        self.res_conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res_bn2b = nn.BatchNorm2d(128)
        self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 3)  # 3 classes: Buy, Hold, Sell
    
    def forward(self, x):
        # Ensure input is in the right shape [batch, channels, height, width]
        if len(x.shape) == 3:  # [batch, height, width]
            x = x.unsqueeze(1)  # Add channel dimension
        elif len(x.shape) == 2:  # [height, width]
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Residual block 1
        residual = self.shortcut1(x)
        x = self.res_conv1a(x)
        x = self.res_bn1a(x)
        x = F.relu(x)
        x = self.res_conv1b(x)
        x = self.res_bn1b(x)
        x = x + residual  # Skip connection
        x = F.relu(x)
        x = self.pool2(x)
        
        # Residual block 2
        residual = self.shortcut2(x)
        x = self.res_conv2a(x)
        x = self.res_bn2a(x)
        x = F.relu(x)
        x = self.res_conv2b(x)
        x = self.res_bn2b(x)
        x = x + residual  # Skip connection
        x = F.relu(x)
        
        # Global average pooling
        x = self.gap(x).squeeze(-1).squeeze(-1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)  # Output probabilities for 3 classes

class CNNTradingModel:
    """
    CNN-based trading model with PyTorch implementation.
    Follows the approach from Sezer & Ozbayoglu (2018).
    """
    
    def __init__(self, input_shape=None, model_dir="models"):
        """
        Initialise the CNN trading model.
        
        Args:
            input_shape: Shape of input data (optional, default is 30x30)
            model_dir: Directory to save trained models
        """
        self.input_shape = input_shape if input_shape else (30, 30)
        self.model_dir = model_dir
        self.model = None
        
        # Modified device selection for cloud environment compatibility
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        except:
            # Fallback to CPU if there are any issues with CUDA detection
            self.device = torch.device("cpu")
            
                
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def build_simple_cnn(self):
        """
        Build a simple CNN model following Sezer & Ozbayoglu (2018).
        This implementation is more robust across different environments.
        
        Returns:
            Compiled CNN model
        """
        # Create a fixed version of SimpleCNN that properly handles channel dimensions
        class FixedSimpleCNN(nn.Module):
            def __init__(self):
                super(FixedSimpleCNN, self).__init__()
                # Input is 1 channel image
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Add padding for consistent dimensions
                self.pool1 = nn.MaxPool2d(kernel_size=2)
                self.dropout1 = nn.Dropout(0.2)
                
                # Second conv layer takes 32 input channels (output from first conv)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Add padding for consistent dimensions
                self.pool2 = nn.MaxPool2d(kernel_size=2)
                self.dropout2 = nn.Dropout(0.2)
                
                # We'll define the fc1 layer with a default reasonable size
                # Based on a 30x30 input, with the above layers, we get 7x7 feature maps
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.dropout3 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(128, 3)  # 3 classes: Buy, Hold, Sell
                
            def forward(self, x):
                # Debug outputs
                initial_shape = x.shape
                
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
                
                # Flatten - handle dynamic input sizes
                x_flat = x.view(x.size(0), -1)
                
                # If fc1 has the wrong input size, recreate it with the correct size
                if self.fc1.in_features != x_flat.shape[1]:
                    device = x.device
                    input_size = x_flat.shape[1]
                    self.fc1 = nn.Linear(input_size, 128).to(device)
                    print(f"Adjusted fc1 layer: {input_size} -> 128")
                
                # Fully connected layers
                x = F.relu(self.fc1(x_flat))
                x = self.dropout3(x)
                x = self.fc2(x)
                
                # Use log softmax for better numerical stability
                return F.softmax(x, dim=1)
        
        # Use the fixed model
        model = FixedSimpleCNN().to(self.device)
        
        # Initialise weights with a robust method
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
        self.model = model
        return model
    
    def build_advanced_cnn(self):
        """
        Build an advanced CNN model with attention mechanisms.
        
        Returns:
            Compiled advanced CNN model
        """
        model = AdvancedCNN().to(self.device)
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, advanced=False, epochs=50, batch_size=32):
        """
        Train the CNN model.
        
        Args:
            X_train: Training data (images)
            y_train: Training labels (0=Hold, 1=Buy, 2=Sell)
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
        
        # Convert labels to one-hot encoding if they're not already
        if len(y_train.shape) == 1 or y_train.shape[1] == 1:
            # Convert labels to integers
            y_train = y_train.astype(int)
            y_val = y_val.astype(int)
            
            # Create one-hot encoded labels
            y_train_one_hot = np.zeros((y_train.shape[0], 3))
            y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
            
            y_val_one_hot = np.zeros((y_val.shape[0], 3))
            y_val_one_hot[np.arange(y_val.shape[0]), y_val] = 1
        else:
            # Already one-hot encoded
            y_train_one_hot = y_train
            y_val_one_hot = y_val
        
        # Call the improved training function
        return self._train_model(
            X_train, y_train_one_hot, X_val, y_val_one_hot,
            epochs=epochs,
            batch_size=batch_size
        )
    
    def _train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Internal method to train the model with proper setup.
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            X_val: Validation data
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimiser
            
        Returns:
            Training history
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        
        # Reshape data if needed to match CNN input requirements
        if len(X_train.shape) == 3:  # [samples, height, width]
            # Already in correct shape for our model
            pass
        elif len(X_train.shape) == 4 and X_train.shape[1] == 1:  # [samples, channels, height, width]
            # Already in correct shape with channel dimension
            pass
        else:
            # Reshape if necessary - this is a fallback
            X_train_tensor = X_train_tensor.reshape(-1, 1, self.input_shape[0], self.input_shape[1])
            X_val_tensor = X_val_tensor.reshape(-1, 1, self.input_shape[0], self.input_shape[1])
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define optimiser
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Define loss function - Cross Entropy for multi-class classification
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        history = {
            'loss': [], 
            'accuracy': [], 
            'val_loss': [], 
            'val_accuracy': [],
            'lr': []
        }
        
        best_val_loss = float('inf')
        patience = 10  # For early stopping
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, torch.max(batch_y, 1)[1])  # CrossEntropyLoss needs class indices
                
                # Backward pass and optimise
                loss.backward()
                optimizer.step()
                
                # Accumulate statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, targets = torch.max(batch_y, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
            
            # Calculate average training metrics
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
                    loss = criterion(outputs, torch.max(batch_y, 1)[1])  # CrossEntropyLoss needs class indices
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, targets = torch.max(batch_y, 1)
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()
            
            # Calculate average validation metrics
            val_loss /= len(val_loader)
            val_accuracy = correct_val / total_val
            
            # Record history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            logger.info(f'Epoch {epoch+1}/{epochs} - '
                  f'loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - '
                  f'val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f} - '
                  f'lr: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Input data (images)
            
        Returns:
            Predicted probabilities for each class [Hold, Buy, Sell]
        """
        if self.model is None:
            raise ValueError("Model has not been built and trained yet")
        
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Reshape data if needed
        if len(X.shape) == 3:  # [samples, height, width]
            # Already in correct shape for our model
            pass
        elif len(X.shape) == 4 and X.shape[1] == 1:  # [samples, channels, height, width]
            # Already in correct shape with channel dimension
            pass
        elif len(X.shape) == 2:  # Single image [height, width]
            X_tensor = X_tensor.unsqueeze(0)  # Add batch dimension
        else:
            # Reshape if necessary - this is a fallback
            X_tensor = X_tensor.reshape(-1, 1, self.input_shape[0], self.input_shape[1])
        
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
    
    def load(self, filename, advanced=False):
        """
        Load a trained model.
        
        Args:
            filename: Filename of the saved model
            advanced: Whether to load the advanced CNN model
        """
        model_path = os.path.join(self.model_dir, f"{filename}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Initialise the model architecture
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
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Ensure y_test is in the right format
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # Convert one-hot encoded labels to class indices
            y_test = np.argmax(y_test, axis=1)
        else:
            y_test = y_test.flatten()
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Calculate other metrics by class
        metrics = {
            'accuracy': accuracy,
            'class_metrics': {}
        }
        
        # Calculate class-specific metrics
        for class_idx in range(3):  # 0=Hold, 1=Buy, 2=Sell
            # Calculate TP, FP, TN, FN for this class
            true_positives = np.sum((y_pred == class_idx) & (y_test == class_idx))
            false_positives = np.sum((y_pred == class_idx) & (y_test != class_idx))
            true_negatives = np.sum((y_pred != class_idx) & (y_test != class_idx))
            false_negatives = np.sum((y_pred != class_idx) & (y_test == class_idx))
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store metrics
            class_name = ['Hold', 'Buy', 'Sell'][class_idx]
            metrics['class_metrics'][class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # Return metrics
        return metrics