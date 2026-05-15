#!/usr/bin/env python3
"""
Scikit-learn Training Pipeline
Complete implementation for traditional ML model training.
"""

import logging
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.feature_extraction.image import extract_patches_2d
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from utils.config import Config
from utils.logger import setup_logger
from utils.metrics import MetricsTracker
from utils.callbacks import CallbackManager
from core.dataset_analyzer import DatasetInfo


class SklearnTrainer:
    """Complete Scikit-learn training pipeline for traditional ML approaches."""
    
    def __init__(self, config: Config):
        """Initialize Scikit-learn trainer."""
        self.config = config
        self.logger = setup_logger(__name__)
        self.metrics_tracker = MetricsTracker()
        
        # Check availability
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Scikit-learn is not installed. Please install with: pip install scikit-learn"
            )
        
        # Initialize training state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.feature_extractor = None
        
        # Initialize callback system
        self.callback_manager = CallbackManager(framework="Scikit-learn")
        self._setup_callbacks()
        
    def prepare_data(self, dataset_info: DatasetInfo, validation_split: float = 0.2) -> Dict[str, Any]:
        """Prepare data for Scikit-learn training by extracting features from images."""
        try:
            self.logger.info("🔄 Preparing Scikit-learn data with feature extraction...")
            
            # Extract features and labels from dataset
            features, labels = self._extract_features_and_labels(dataset_info)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(labels)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, y_encoded, 
                test_size=validation_split, 
                random_state=42, 
                stratify=y_encoded
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Store data
            self.X_train = X_train_scaled
            self.X_val = X_val_scaled  
            self.y_train = y_train
            self.y_val = y_val
            
            # Prepare data info
            data_info = {
                'feature_dim': X_train_scaled.shape[1],
                'num_classes': len(np.unique(y_encoded)),
                'train_samples': len(X_train_scaled),
                'val_samples': len(X_val_scaled),
                'class_names': list(self.label_encoder.classes_)
            }
            
            self.logger.info(f"✅ Data prepared - Features: {data_info['feature_dim']}, Train: {data_info['train_samples']}, Val: {data_info['val_samples']}")
            return data_info
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            raise
    
    def _extract_features_and_labels(self, dataset_info: DatasetInfo) -> Tuple[np.ndarray, List[str]]:
        """Extract features from images for traditional ML."""
        try:
            features = []
            labels = []
            
            dataset_path = Path(dataset_info.dataset_path)
            
            # Process each class directory
            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    self.logger.info(f"📁 Processing class: {class_name}")
                    
                    # Process images in class directory
                    image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
                                 list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.bmp"))
                    
                    for img_path in image_files[:500]:  # Limit for performance
                        try:
                            # Extract features from image
                            img_features = self._extract_image_features(str(img_path), dataset_info.image_size)
                            if img_features is not None:
                                features.append(img_features)
                                labels.append(class_name)
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to process {img_path}: {e}")
                            continue
            
            if not features:
                raise ValueError("No features extracted from dataset")
            
            features_array = np.array(features)
            self.logger.info(f"✅ Extracted {len(features_array)} feature vectors of dimension {features_array.shape[1]}")
            
            return features_array, labels
            
        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            raise
    
    def _extract_image_features(self, image_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract traditional ML features from an image."""
        try:
            # Try different image loading methods
            img = None
            
            if CV2_AVAILABLE:
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is None and PIL_AVAILABLE:
                with Image.open(image_path) as pil_img:
                    img = pil_img.resize(target_size)
                    img = np.array(img)
            
            if img is None:
                return None
            
            # Ensure grayscale for simpler features
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            
            # Extract multiple types of features
            features = []
            
            # 1. Histogram features
            hist = np.histogram(img.flatten(), bins=32, range=[0, 256])[0]
            features.extend(hist / np.sum(hist))  # Normalize
            
            # 2. Statistical features
            features.extend([
                np.mean(img),
                np.std(img),
                np.min(img),
                np.max(img),
                np.median(img)
            ])
            
            # 3. Texture features (simple)
            if CV2_AVAILABLE and img.shape[0] > 1 and img.shape[1] > 1:
                # Sobel edges
                sobel_x = cv2.Sobel(img.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
                edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                
                features.extend([
                    np.mean(edge_magnitude),
                    np.std(edge_magnitude)
                ])
            else:
                features.extend([0.0, 0.0])  # Placeholder
            
            # 4. Downsampled pixel features (for small images)
            if img.size <= 1024:  # Only for small images
                downsampled = cv2.resize(img.astype(np.uint8), (8, 8)) if CV2_AVAILABLE else img[:8, :8]
                features.extend(downsampled.flatten() / 255.0)
            else:
                # Use patches for larger images
                try:
                    patches = extract_patches_2d(img, (8, 8), max_patches=16, random_state=42)
                    patch_features = np.mean(patches.reshape(patches.shape[0], -1), axis=0)
                    features.extend(patch_features / 255.0)
                except:
                    # Fallback: just resize
                    if CV2_AVAILABLE:
                        downsampled = cv2.resize(img.astype(np.uint8), (8, 8))
                        features.extend(downsampled.flatten() / 255.0)
                    else:
                        features.extend([0.0] * 64)  # Placeholder
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature extraction failed for {image_path}: {e}")
            return None
    
    def build_model(self, model_config: Dict[str, Any], data_info: Dict[str, Any]):
        """Build Scikit-learn model."""
        try:
            architecture = model_config.get('architecture', 'Random Forest')
            self.logger.info(f"🏗️ Building {architecture} model...")
            
            if architecture == 'Random Forest':
                self.model = RandomForestClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    max_depth=model_config.get('max_depth', 10),
                    random_state=42,
                    n_jobs=-1
                )
            elif architecture == 'Support Vector Machine':
                self.model = SVC(
                    C=model_config.get('C', 1.0),
                    kernel=model_config.get('kernel', 'rbf'),
                    random_state=42,
                    probability=True
                )
            elif architecture == 'Gradient Boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    learning_rate=model_config.get('learning_rate', 0.1),
                    max_depth=model_config.get('max_depth', 3),
                    random_state=42
                )
            elif architecture == 'Logistic Regression':
                self.model = LogisticRegression(
                    C=model_config.get('C', 1.0),
                    random_state=42,
                    max_iter=1000
                )
            elif architecture == 'Neural Network':
                self.model = MLPClassifier(
                    hidden_layer_sizes=model_config.get('hidden_layers', (100, 50)),
                    learning_rate_init=model_config.get('learning_rate', 0.001),
                    max_iter=model_config.get('max_iter', 200),
                    random_state=42
                )
            elif architecture == 'K-Nearest Neighbors':
                self.model = KNeighborsClassifier(
                    n_neighbors=model_config.get('n_neighbors', 5),
                    weights=model_config.get('weights', 'uniform')
                )
            else:
                # Default to Random Forest
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            
            self.logger.info(f"✅ {architecture} model built successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Model building failed: {e}")
            raise
    
    def _setup_callbacks(self):
        """Setup Scikit-learn training callbacks."""
        from utils.callbacks import (
            EarlyStopping, ModelCheckpoint, ProgressBar, 
            MetricsLogger, PerformanceMonitor
        )
        
        # Progress bar
        self.callback_manager.add_callback(ProgressBar(verbose=True))
        
        # Performance monitoring
        self.callback_manager.add_callback(PerformanceMonitor(verbose=True))
        
        # Metrics logger will be added in train() with proper save directory
        
    def train(self, epochs: int = 1, save_model: bool = True, 
              model_save_dir: str = "./experiments") -> Dict[str, Any]:
        """Train the Scikit-learn model with callbacks."""
        try:
            self.logger.info("🚀 Starting Scikit-learn training...")
            
            if self.model is None or self.X_train is None:
                raise ValueError("Model and data must be prepared first")
            
            # Create save directory
            save_dir = Path(model_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Add metrics logger with save directory
            from utils.callbacks import MetricsLogger, ModelCheckpoint
            metrics_logger = MetricsLogger(save_dir, framework="Scikit-learn")
            self.callback_manager.add_callback(metrics_logger)
            
            # Add model checkpoint
            if save_model:
                checkpoint_path = save_dir / "sklearn_best_model.joblib"
                model_checkpoint = ModelCheckpoint(
                    checkpoint_path, 
                    monitor='val_acc', 
                    save_best_only=True, 
                    mode='max'
                )
                model_checkpoint.set_model(self.model)
                self.callback_manager.add_callback(model_checkpoint)
            
            # Start training timer
            training_start_time = time.time()
            
            # Initialize metrics tracker
            self.metrics_tracker.reset()
            
            # Setup callbacks
            self._setup_callbacks()
            
            # Callback: Training begins
            self.callback_manager.on_train_begin({
                'total_epochs': 1,
                'framework': 'Scikit-learn',
                'model_type': self.model.__class__.__name__
            })
            
            # Callback: Epoch begins (Scikit-learn trains in one epoch)
            self.callback_manager.on_epoch_begin(0, {'total_epochs': 1})
            
            # Train the model
            self.model.fit(self.X_train, self.y_train)
            
            # Calculate training time
            training_time = time.time() - training_start_time
            
            # Make predictions
            y_train_pred = self.model.predict(self.X_train)
            y_val_pred = self.model.predict(self.X_val)
            
            # Calculate accuracies
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            val_accuracy = accuracy_score(self.y_val, y_val_pred)
            
            # Cross-validation for robust evaluation
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
            
            # Track metrics
            epoch_metrics = {
                'train_loss': 1 - train_accuracy,  # Convert to loss
                'train_acc': train_accuracy,
                'train_accuracy': train_accuracy,
                'val_loss': 1 - val_accuracy,
                'val_acc': val_accuracy,
                'val_accuracy': val_accuracy
            }
            
            # Enhanced metrics tracking
            accuracy_improvement = self.metrics_tracker.track_accuracy(1, epoch_metrics)
            
            # Callback: Epoch ends
            self.callback_manager.on_epoch_end(0, epoch_metrics)
            
            # Prepare comprehensive results
            results = {
                'best_accuracy': float(val_accuracy),
                'best_loss': float(1 - val_accuracy),
                'final_train_accuracy': float(train_accuracy),
                'final_val_accuracy': float(val_accuracy),
                'cross_val_mean': float(np.mean(cv_scores)),
                'cross_val_std': float(np.std(cv_scores)),
                'training_time': training_time,
                'total_epochs': 1,  # Scikit-learn trains in one go
                'framework': 'Scikit-learn',
                'model_architecture': self.model.__class__.__name__,
                'overfitting_detected': abs(train_accuracy - val_accuracy) > 0.1,
                'convergence_status': 'completed',
                'feature_importance': self._get_feature_importance()
            }
            
            # Generate detailed evaluation
            classification_rep = classification_report(
                self.y_val, y_val_pred, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            results['classification_report'] = classification_rep
            
            # Save model if requested
            if save_model:
                # Generate intelligent model names with framework, architecture, dataset, and timestamp
                from utils.model_naming import generate_comprehensive_model_paths, extract_dataset_name
                
                # Extract dataset name from session state or config
                dataset_name = "unknown"
                if hasattr(self, 'dataset_path') and self.dataset_path:
                    dataset_name = extract_dataset_name(dataset_path=self.dataset_path)
                elif hasattr(self, 'config') and hasattr(self.config, 'dataset_path'):
                    dataset_name = extract_dataset_name(dataset_path=self.config.dataset_path)
                
                # Get architecture name
                architecture = self.model.__class__.__name__ if self.model else "unknown"
                
                # Generate comprehensive paths
                model_paths = generate_comprehensive_model_paths(
                    base_dir=save_dir,
                    framework="scikit-learn",
                    architecture=architecture,
                    dataset_name=dataset_name,
                    task_type="classification"  # Most sklearn models are classification
                )
                
                # Save model with intelligent naming
                model_path = model_paths['model']
                scaler_path = save_dir / f"{model_paths['base_name']}_scaler.joblib"
                encoder_path = save_dir / f"{model_paths['base_name']}_encoder.joblib"
                
                joblib.dump(self.model, model_path)
                joblib.dump(self.scaler, scaler_path)
                joblib.dump(self.label_encoder, encoder_path)
                
                results['model_path'] = str(model_path)
                results['scaler_path'] = str(scaler_path)
                results['encoder_path'] = str(encoder_path)
                results['base_name'] = model_paths['base_name']
                
                self.logger.info(f"💾 Model saved to: {model_path}")
                self.logger.info(f"💾 Scaler saved to: {scaler_path}")
                self.logger.info(f"💾 Encoder saved to: {encoder_path}")
            
            # Save results
            results_path = save_dir / "sklearn_training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Callback: Training ends
            self.callback_manager.on_train_end(results)
            
            self.logger.info(f"🎉 Scikit-learn training completed!")
            self.logger.info(f"   📊 Validation Accuracy: {val_accuracy:.6f}")
            self.logger.info(f"   📊 Cross-val Mean: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
            self.logger.info(f"   ⏱️ Training Time: {training_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Scikit-learn training failed: {e}")
            raise
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                # Return top 10 feature indices and their importance
                top_indices = np.argsort(importances)[-10:][::-1]
                return {
                    f'feature_{idx}': float(importances[idx]) 
                    for idx in top_indices
                }
            elif hasattr(self.model, 'coef_'):
                # For linear models
                coefs = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
                top_indices = np.argsort(coefs)[-10:][::-1]
                return {
                    f'feature_{idx}': float(coefs[idx]) 
                    for idx in top_indices
                }
            else:
                return None
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract feature importance: {e}")
            return None
    
    def evaluate(self, test_data_path: Optional[str] = None) -> Dict[str, float]:
        """Evaluate the trained model."""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet. Call train() first.")
            
            # Use validation data for evaluation
            y_pred = self.model.predict(self.X_val)
            y_pred_proba = self.model.predict_proba(self.X_val) if hasattr(self.model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_val, y_pred)
            
            # Classification report
            report = classification_report(
                self.y_val, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            eval_results = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
            self.logger.info(f"✅ Evaluation completed: {eval_results}")
            return eval_results
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            return {}
    
    def save_model(self, model_path: str):
        """Save the trained Scikit-learn model."""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train a model first.")
            
            # Save model, scaler, and encoder
            base_path = Path(model_path).with_suffix('')
            
            joblib.dump(self.model, f"{base_path}_model.joblib")
            joblib.dump(self.scaler, f"{base_path}_scaler.joblib") 
            joblib.dump(self.label_encoder, f"{base_path}_encoder.joblib")
            
            self.logger.info(f"💾 Model components saved to: {base_path}_*.joblib")
            
        except Exception as e:
            self.logger.error(f"❌ Model saving failed: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Load a saved Scikit-learn model."""
        try:
            base_path = Path(model_path).with_suffix('')
            
            self.model = joblib.load(f"{base_path}_model.joblib")
            self.scaler = joblib.load(f"{base_path}_scaler.joblib")
            self.label_encoder = joblib.load(f"{base_path}_encoder.joblib")
            
            self.logger.info(f"📂 Model loaded from: {base_path}_*.joblib")
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            raise


def create_sklearn_trainer(config: Config) -> SklearnTrainer:
    """Factory function to create Scikit-learn trainer."""
    return SklearnTrainer(config)