#!/usr/bin/env python3
"""
YOLO Trainer - Standard Ultralytics Pipeline
Complete implementation using the official ultralytics YOLO API.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

from utils.config import Config
from utils.logger import setup_logger


class YOLOTrainer:
    """
    Standard YOLO trainer using ultralytics library.
    Supports YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x variants.
    """
    
    YOLO_VARIANTS = {
        "yolov8n": {"model": "yolov8n.pt", "params": "3.2M", "speed": "fastest"},
        "yolov8s": {"model": "yolov8s.pt", "params": "11.2M", "speed": "fast"},
        "yolov8m": {"model": "yolov8m.pt", "params": "25.9M", "speed": "balanced"},
        "yolov8l": {"model": "yolov8l.pt", "params": "43.7M", "speed": "accurate"},
        "yolov8x": {"model": "yolov8x.pt", "params": "68.2M", "speed": "most accurate"},
    }
    
    def __init__(self, config: Config, yolo_variant: str = "yolov8s"):
        """
        Initialize YOLO trainer.
        
        Args:
            config: Training configuration
            yolo_variant: YOLO variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        """
        self.config = config
        
        # Check if yolo_variant is in config_params
        if hasattr(config, 'model_config') and hasattr(config.model_config, 'config_params'):
            config_variant = config.model_config.config_params.get('yolo_variant')
            if config_variant:
                yolo_variant = config_variant
        
        self.yolo_variant = yolo_variant.lower()
        self.logger = setup_logger(__name__)
        
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics not installed. Install with: pip install ultralytics"
            )
        
        if self.yolo_variant not in self.YOLO_VARIANTS:
            self.logger.warning(
                f"Unknown YOLO variant '{yolo_variant}'. Using yolov8s instead."
            )
            self.yolo_variant = "yolov8s"
        
        self.model = None
        self.results = None
        
    def prepare_data(self) -> str:
        """
        Prepare data in YOLO format.
        Creates a data.yaml file for ultralytics.
        
        Returns:
            Path to data.yaml file
        """
        import yaml
        
        dataset_path = Path(self.config.dataset_info.dataset_path)
        data_yaml_path = self.config.output_dir / "data.yaml"
        
        # Check if dataset already has data.yaml
        existing_yaml = dataset_path / "data.yaml"
        if existing_yaml.exists():
            self.logger.info(f"✅ Found existing data.yaml at {existing_yaml}")
            return str(existing_yaml)
        
        # Create data.yaml for YOLO training
        num_classes = self.config.dataset_info.num_classes
        class_names = self.config.dataset_info.class_names
        
        if not class_names:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        # YOLO data.yaml structure
        data_config = {
            "path": str(dataset_path.absolute()),
            "train": "images/train" if (dataset_path / "images" / "train").exists() else "train/images",
            "val": "images/val" if (dataset_path / "images" / "val").exists() else "val/images",
            "nc": num_classes,
            "names": class_names
        }
        
        # Save data.yaml
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        self.logger.info(f"📝 Created data.yaml at {data_yaml_path}")
        self.logger.info(f"   Classes: {num_classes}, Samples: {self.config.dataset_info.num_samples}")
        
        return str(data_yaml_path)
    
    def build_model(self):
        """
        Build YOLO model.
        
        Returns:
            YOLO model instance
        """
        variant_info = self.YOLO_VARIANTS[self.yolo_variant]
        model_name = variant_info["model"]
        
        self.logger.info(
            f"🏗️ Building {self.yolo_variant.upper()} "
            f"({variant_info['params']} params, {variant_info['speed']})"
        )
        
        # Load pretrained model
        self.model = YOLO(model_name)
        
        self.logger.info(f"✅ Model loaded: {model_name}")
        return self.model
    
    def train(self, save_model: bool = True) -> Dict[str, Any]:
        """
        Train YOLO model using ultralytics standard pipeline.
        
        Args:
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info(f"🚀 Starting YOLO Training: {self.yolo_variant.upper()}")
            self.logger.info("=" * 80)
            
            start_time = time.time()
            
            # Prepare data
            data_yaml = self.prepare_data()
            
            # Build model
            if self.model is None:
                self.build_model()
            
            # Training parameters
            epochs = self.config.max_epochs
            batch_size = self.config.batch_size
            imgsz = 640  # Standard YOLO input size
            
            # Extract image size from config if available
            if hasattr(self.config.model_config, 'input_size'):
                input_size = self.config.model_config.input_size
                if isinstance(input_size, (tuple, list)) and len(input_size) >= 2:
                    imgsz = input_size[0] if isinstance(input_size[0], int) else 640
            
            # Learning rate
            lr0 = self.config.learning_rate
            
            # Output directory
            project = str(self.config.output_dir.parent)
            name = self.config.output_dir.name
            
            self.logger.info(f"📊 Training Configuration:")
            self.logger.info(f"   Epochs: {epochs}")
            self.logger.info(f"   Batch Size: {batch_size}")
            self.logger.info(f"   Image Size: {imgsz}")
            self.logger.info(f"   Learning Rate: {lr0}")
            self.logger.info(f"   Output: {project}/{name}")
            
            # Train using ultralytics standard API
            self.results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                lr0=lr0,
                project=project,
                name=name,
                exist_ok=True,
                pretrained=True,
                verbose=True,
                plots=True,
                save=save_model,
                save_period=max(1, epochs // 10),  # Save checkpoint every 10%
                device="cuda:0" if self._has_cuda() else "cpu",
                workers=4,
                patience=self.config.early_stopping_patience if self.config.early_stopping_patience > 0 else 50,
            )
            
            training_time = time.time() - start_time
            
            # Extract metrics from results
            metrics = self._extract_metrics()
            
            # Prepare results
            model_path = self.config.output_dir / "weights" / "best.pt"
            
            results = {
                "model_path": str(model_path) if model_path.exists() else "",
                "final_metrics": metrics,
                "training_time": training_time,
                "epochs_completed": epochs,
                "framework": "ultralytics-yolo",
                "variant": self.yolo_variant,
                "training_history": self._get_training_history(),
            }
            
            self.logger.info("=" * 80)
            self.logger.info("✅ Training Complete!")
            self.logger.info(f"⏱️  Total Time: {training_time/60:.2f} minutes")
            self.logger.info(f"📊 Final mAP50: {metrics.get('map50', 0):.4f}")
            self.logger.info(f"📊 Final mAP50-95: {metrics.get('map50_95', 0):.4f}")
            self.logger.info(f"💾 Model saved to: {model_path}")
            self.logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _extract_metrics(self) -> Dict[str, float]:
        """Extract final metrics from YOLO results."""
        if self.results is None:
            return {}
        
        try:
            # Ultralytics results object has metrics
            metrics = {}
            
            if hasattr(self.results, 'results_dict'):
                res_dict = self.results.results_dict
                metrics = {
                    "map50": res_dict.get("metrics/mAP50(B)", 0.0),
                    "map50_95": res_dict.get("metrics/mAP50-95(B)", 0.0),
                    "precision": res_dict.get("metrics/precision(B)", 0.0),
                    "recall": res_dict.get("metrics/recall(B)", 0.0),
                }
            
            # Try to read from CSV if available
            csv_path = self.config.output_dir / "results.csv"
            if csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    metrics["map50"] = last_row.get("metrics/mAP50(B)", 0.0)
                    metrics["map50_95"] = last_row.get("metrics/mAP50-95(B)", 0.0)
                    metrics["precision"] = last_row.get("metrics/precision(B)", 0.0)
                    metrics["recall"] = last_row.get("metrics/recall(B)", 0.0)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Could not extract metrics: {e}")
            return {}
    
    def _get_training_history(self) -> Dict[str, list]:
        """Get training history from results CSV."""
        try:
            csv_path = self.config.output_dir / "results.csv"
            if not csv_path.exists():
                return {}
            
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            history = {
                "epoch": df["epoch"].tolist() if "epoch" in df else [],
                "train_loss": df["train/box_loss"].tolist() if "train/box_loss" in df else [],
                "val_map50": df["metrics/mAP50(B)"].tolist() if "metrics/mAP50(B)" in df else [],
                "val_map50_95": df["metrics/mAP50-95(B)"].tolist() if "metrics/mAP50-95(B)" in df else [],
            }
            
            return history
            
        except Exception as e:
            self.logger.warning(f"Could not read training history: {e}")
            return {}
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation on the trained model.
        
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        self.logger.info("🔍 Running validation...")
        
        data_yaml = self.prepare_data()
        val_results = self.model.val(data=data_yaml)
        
        metrics = {
            "map50": float(val_results.box.map50),
            "map50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        }
        
        self.logger.info(f"✅ Validation complete: mAP50={metrics['map50']:.4f}")
        
        return metrics
