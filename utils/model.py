"""
CNN Model Module

This module implements CNN models for trading signal prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def init_cnn_weights(module):
    """Initialize CNN weights (shared utility)."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class SimpleCNN(nn.Module):
    """Optimised CNN designed to prevent overfitting."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Optimised architecture
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.6)
        
        # Smaller FC layers
        self.fc1 = nn.Linear(24 * 15 * 15, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(64, 3)
        
        # Initialise weights
        self.apply(init_cnn_weights)
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class AdvancedCNN(nn.Module):
    """Advanced CNN with more layers."""
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        
        # More complex architecture
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.6)
        
        # Larger FC layers
        self.fc1 = nn.Linear(32 * 15 * 15, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(128, 3)
        
        # Initialise weights
        self.apply(init_cnn_weights)
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class CNNTradingModel:
    """CNN trading model wrapper."""
    
    def __init__(self, input_shape=(30, 30), model_dir="models"):
        self.input_shape = input_shape
        self.model_dir = model_dir
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def build_simple_cnn(self):
        """Build optimised simple CNN."""
        self.model = SimpleCNN().to(self.device)
        return self.model
    
    def build_advanced_cnn(self):
        """Build advanced CNN."""
        self.model = AdvancedCNN().to(self.device)
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=50, batch_size=32, learning_rate=0.001, 
            weight_decay=1e-4, patience=8, 
            use_scheduler=True, use_early_stopping=True,
            use_gradient_clipping=True, gradient_clip_max_norm=1.0,
            scheduler_factor=0.5, scheduler_patience=3,
            early_stopping_min_epochs=20,
            callback=None):
        """Train the CNN model."""
        
        if self.model is None:
            raise ValueError("Model has not been built yet")
        
        # Convert labels to class indices if needed
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train = np.argmax(y_train, axis=1)
            if y_val is not None:
                y_val = np.argmax(y_val, axis=1)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        
        # Handle validation data
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
        else:
            # Create temporal validation split
            split_idx = int(0.8 * len(X_train_tensor))
            X_val_tensor = X_train_tensor[split_idx:]
            y_val_tensor = y_train_tensor[split_idx:]
            X_train_tensor = X_train_tensor[:split_idx]
            y_train_tensor = y_train_tensor[:split_idx]
        
        # Ensure correct shape
        if len(X_train_tensor.shape) == 3:
            X_train_tensor = X_train_tensor.unsqueeze(1)
            X_val_tensor = X_val_tensor.unsqueeze(1)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Optimiser
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Scheduler
        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
            )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training history
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                if use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_max_norm)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = correct / total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # Early stopping
            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
            
            # Store history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Callback
            if callback is not None:
                callback(epoch, train_loss, train_accuracy, val_loss, val_accuracy)
            
            # Early stopping check
            if use_early_stopping and patience_counter >= patience and epoch > early_stopping_min_epochs:
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model has not been built and trained yet")
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        if len(X.shape) == 3:
            X_tensor = X_tensor.unsqueeze(1)
        elif len(X.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = F.softmax(logits, dim=1)
            predictions = probabilities.cpu().numpy()
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model with comprehensive metrics."""
        if self.model is None:
            raise ValueError("Model has not been built and trained yet")
        
        y_pred_prob = self.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
        else:
            y_test = y_test.flatten()
        
        accuracy = np.mean(y_pred == y_test)
        
        metrics = {
            'accuracy': accuracy,
            'num_samples': len(y_test),
            'class_metrics': {}
        }
        
        # Calculate class-specific metrics
        for class_idx in range(3):
            class_mask = (y_test == class_idx)
            if np.sum(class_mask) > 0:
                tp = np.sum((y_pred == class_idx) & (y_test == class_idx))
                fp = np.sum((y_pred == class_idx) & (y_test != class_idx))
                tn = np.sum((y_pred != class_idx) & (y_test != class_idx))
                fn = np.sum((y_pred != class_idx) & (y_test == class_idx))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                class_name = ['Hold', 'Buy', 'Sell'][class_idx]
                metrics['class_metrics'][class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': np.sum(class_mask)
                }
        
        return metrics
    
    def save(self, filename):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = os.path.join(self.model_dir, f"{filename}.pt")
        torch.save(self.model.state_dict(), model_path)
    
    def load(self, filename, architecture='simple'):
        """Load a trained model."""
        model_path = os.path.join(self.model_dir, f"{filename}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Build appropriate architecture
        if architecture == 'simple':
            self.build_simple_cnn()
        else:
            self.build_advanced_cnn()
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        return self.model