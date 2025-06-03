"""
CNN Model Module - Cleaned Version

Only 2 models: Simple (prevents overfitting) and Advanced (more complex)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """Optimised CNN designed to prevent overfitting while maintaining good performance."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # OPTIMISED architecture based on model results analysis
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)  # Reduced filters
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)  # Progressive increase
        self.bn2 = nn.BatchNorm2d(24)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.6)  # Increased regularisation
        
        # Smaller FC layers to prevent overfitting
        self.fc1 = nn.Linear(24 * 15 * 15, 64)  # Much smaller
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)  # Higher dropout
        self.fc2 = nn.Linear(64, 3)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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
    """Advanced CNN with more layers but still stable."""
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
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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
    """Clean CNN trading model with only 2 architectures."""
    
    def __init__(self, input_shape=(30, 30), model_dir="models"):
        self.input_shape = input_shape
        self.model_dir = model_dir
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def build_simple_cnn(self):
        """Build optimised simple CNN - recommended for most use cases."""
        self.model = SimpleCNN().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Optimised Simple CNN model created with {total_params} parameters")
        return self.model
    
    def build_advanced_cnn(self):
        """Build advanced CNN - more complex architecture."""
        self.model = AdvancedCNN().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Advanced CNN model created with {total_params} parameters")
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=50, batch_size=32, learning_rate=0.001, 
            weight_decay=1e-4, patience=5, 
            use_scheduler=True, use_early_stopping=True,
            use_gradient_clipping=True, gradient_clip_max_norm=1.0,  
            scheduler_factor=0.5, scheduler_patience=3,             
            early_stopping_min_epochs=20,                           
            callback=None):
        
        """Unified training method for both models."""
        
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
            # Use provided validation data
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
            logger.info("Using provided validation data")
        else:
            # Create temporal validation split from training data (no shuffling)
            split_idx = int(0.8 * len(X_train_tensor))
            X_val_tensor = X_train_tensor[split_idx:]
            y_val_tensor = y_train_tensor[split_idx:]
            X_train_tensor = X_train_tensor[:split_idx]
            y_train_tensor = y_train_tensor[:split_idx]
            logger.info("Created temporal validation split from training data")
        
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
        
        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # CONDITIONAL SCHEDULER - only create if use_scheduler is True
        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
            )
            logger.info("Learning rate scheduler enabled")
        else:
            logger.info("Learning rate scheduler disabled")
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training history
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Training samples: {len(X_train_tensor)}, Validation samples: {len(X_val_tensor)}")
        logger.info(f"Early stopping: {'Enabled' if use_early_stopping else 'Disabled'} (patience: {patience}, min_epochs: {early_stopping_min_epochs})")
        logger.info(f"Learning rate scheduler: {'Enabled' if use_scheduler else 'Disabled'} (factor: {scheduler_factor}, patience: {scheduler_patience})")
        logger.info(f"Gradient clipping: {'Enabled' if use_gradient_clipping else 'Disabled'} (max_norm: {gradient_clip_max_norm})")
        
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
            
            # CONDITIONAL LEARNING RATE SCHEDULING
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # CONDITIONAL EARLY STOPPING LOGIC
            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
            else:
                # If early stopping disabled, still track best model but don't increment patience
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                patience_counter = 0  # Keep patience at 0 when early stopping disabled
            
            # Store history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Callback
            if callback is not None:
                callback(epoch, train_loss, train_accuracy, val_loss, val_accuracy)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
                print(f'  Learning Rate: {current_lr:.6f}')
                if use_early_stopping:
                    print(f'  Patience: {patience_counter}/{patience}')
                else:
                    print(f'  Early Stopping: Disabled')
            
            # CONDITIONAL EARLY STOPPING
            if use_early_stopping and patience_counter >= patience and epoch > early_stopping_min_epochs:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Restore best model if we saved one
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with validation loss: {best_val_loss:.4f}")
        
        # Final training summary
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        overfitting_gap = final_train_acc - final_val_acc
        
        logger.info("Training completed!")
        logger.info(f"Final training accuracy: {final_train_acc:.4f}")
        logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        logger.info(f"Overfitting gap: {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.15:
            logger.warning("Significant overfitting detected - consider increasing regularization")
        
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
        logger.info(f"Model saved to {model_path}")
    
    def load(self, filename):
        """Load a trained model."""
        model_path = os.path.join(self.model_dir, f"{filename}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Need to know which architecture to load
        # This is a limitation - you'd need to specify which model type
        self.build_simple_cnn()  # Default to simple
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
        
        return self.model