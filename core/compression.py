"""
Model Compression and Quantization for VisionForge v2.1

Reduces model size and improves inference speed:
- Post-training quantization (INT8, FP16)
- Quantization-aware training
- Model pruning (structured and unstructured)
- Knowledge distillation
- ONNX export and optimization
- Mobile deployment optimization (TFLite, CoreML)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.quantization as torch_quant
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelCompressor:
    """
    Main interface for model compression.
    Supports PyTorch and TensorFlow models.
    """
    
    def __init__(self, model, framework: str):
        """
        Initialize model compressor.
        
        Args:
            model: Model to compress
            framework: 'pytorch' or 'tensorflow'
        """
        self.model = model
        self.framework = framework.lower()
        
        if self.framework not in ['pytorch', 'tensorflow']:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def quantize(
        self,
        method: str = 'dynamic',
        dtype: str = 'int8',
        calibration_data: Optional[Any] = None
    ) -> Any:
        """
        Quantize model to reduce size and improve speed.
        
        Args:
            method: Quantization method ('dynamic', 'static', 'qat')
            dtype: Target data type ('int8', 'fp16')
            calibration_data: Data for calibration (required for static quantization)
            
        Returns:
            Quantized model
        """
        if self.framework == 'pytorch':
            return self._quantize_pytorch(method, dtype, calibration_data)
        else:
            return self._quantize_tensorflow(method, dtype, calibration_data)
    
    def _quantize_pytorch(
        self,
        method: str,
        dtype: str,
        calibration_data: Optional[Any]
    ) -> nn.Module:
        """Quantize PyTorch model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if method == 'dynamic':
            # Dynamic quantization (activations quantized on-the-fly)
            quantized_model = torch_quant.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8 if dtype == 'int8' else torch.float16
            )
            logger.info("Applied dynamic quantization")
            
        elif method == 'static':
            # Static quantization (requires calibration)
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            
            # Prepare model for quantization
            self.model.eval()
            self.model.qconfig = torch_quant.get_default_qconfig('fbgemm')
            model_prepared = torch_quant.prepare(self.model)
            
            # Calibrate
            logger.info("Calibrating model...")
            with torch.no_grad():
                for batch in calibration_data:
                    model_prepared(batch)
            
            # Convert to quantized model
            quantized_model = torch_quant.convert(model_prepared)
            logger.info("Applied static quantization")
            
        elif method == 'qat':
            # Quantization-aware training
            self.model.qconfig = torch_quant.get_default_qat_qconfig('fbgemm')
            model_prepared = torch_quant.prepare_qat(self.model)
            logger.info("Prepared model for QAT (training required)")
            quantized_model = model_prepared
            
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        return quantized_model
    
    def _quantize_tensorflow(
        self,
        method: str,
        dtype: str,
        calibration_data: Optional[Any]
    ) -> tf.keras.Model:
        """Quantize TensorFlow model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        if method == 'dynamic' or method == 'static':
            # Post-training quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            if dtype == 'fp16':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            else:  # int8
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                if calibration_data is not None:
                    # Representative dataset for full integer quantization
                    def representative_dataset():
                        for batch in calibration_data:
                            yield [batch]
                    
                    converter.representative_dataset = representative_dataset
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.uint8
                    converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            logger.info(f"Applied {dtype} quantization")
            
            return tflite_model
        
        elif method == 'qat':
            # Quantization-aware training
            import tensorflow_model_optimization as tfmot
            
            quantize_model = tfmot.quantization.keras.quantize_model
            quantized_model = quantize_model(self.model)
            logger.info("Prepared model for QAT")
            
            return quantized_model
        
        else:
            raise ValueError(f"Unknown quantization method: {method}")
    
    def prune(
        self,
        pruning_ratio: float = 0.5,
        method: str = 'magnitude',
        structured: bool = False
    ) -> Any:
        """
        Prune model to reduce parameters.
        
        Args:
            pruning_ratio: Fraction of parameters to prune
            method: Pruning method ('magnitude', 'random', 'l1')
            structured: Use structured pruning
            
        Returns:
            Pruned model
        """
        if self.framework == 'pytorch':
            return self._prune_pytorch(pruning_ratio, method, structured)
        else:
            return self._prune_tensorflow(pruning_ratio, method, structured)
    
    def _prune_pytorch(
        self,
        pruning_ratio: float,
        method: str,
        structured: bool
    ) -> nn.Module:
        """Prune PyTorch model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        import torch.nn.utils.prune as prune
        
        # Apply pruning to conv and linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if structured:
                    # Structured pruning (prune entire filters/neurons)
                    if isinstance(module, nn.Conv2d):
                        prune.ln_structured(
                            module,
                            name='weight',
                            amount=pruning_ratio,
                            n=2,
                            dim=0  # Prune output channels
                        )
                else:
                    # Unstructured pruning
                    if method == 'magnitude':
                        prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                    elif method == 'random':
                        prune.random_unstructured(module, name='weight', amount=pruning_ratio)
                    else:
                        raise ValueError(f"Unknown pruning method: {method}")
        
        logger.info(f"Applied {pruning_ratio*100}% {method} pruning")
        return self.model
    
    def _prune_tensorflow(
        self,
        pruning_ratio: float,
        method: str,
        structured: bool
    ) -> tf.keras.Model:
        """Prune TensorFlow model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        try:
            import tensorflow_model_optimization as tfmot
        except ImportError:
            raise ImportError("tensorflow-model-optimization required for pruning")
        
        # Define pruning schedule
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=pruning_ratio,
                begin_step=0,
                end_step=1000
            )
        }
        
        # Apply pruning
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            **pruning_params
        )
        
        logger.info(f"Applied {pruning_ratio*100}% pruning")
        return model_for_pruning
    
    def export_onnx(
        self,
        save_path: Path,
        input_shape: Tuple[int, ...],
        opset_version: int = 13
    ):
        """
        Export model to ONNX format.
        
        Args:
            save_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
        """
        if self.framework == 'pytorch':
            self._export_onnx_pytorch(save_path, input_shape, opset_version)
        else:
            self._export_onnx_tensorflow(save_path, input_shape, opset_version)
    
    def _export_onnx_pytorch(
        self,
        save_path: Path,
        input_shape: Tuple[int, ...],
        opset_version: int
    ):
        """Export PyTorch model to ONNX."""
        if not TORCH_AVAILABLE or not ONNX_AVAILABLE:
            raise ImportError("PyTorch and ONNX required")
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Exported ONNX model to {save_path}")
    
    def _export_onnx_tensorflow(
        self,
        save_path: Path,
        input_shape: Tuple[int, ...],
        opset_version: int
    ):
        """Export TensorFlow model to ONNX."""
        try:
            import tf2onnx
        except ImportError:
            raise ImportError("tf2onnx required for TensorFlow to ONNX conversion")
        
        # Convert
        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
        output_path = str(save_path)
        
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model,
            input_signature=spec,
            opset=opset_version,
            output_path=output_path
        )
        
        logger.info(f"Exported ONNX model to {save_path}")
    
    def get_model_size(self) -> Dict[str, float]:
        """
        Get model size information.
        
        Returns:
            Dictionary with size metrics
        """
        if self.framework == 'pytorch':
            return self._get_size_pytorch()
        else:
            return self._get_size_tensorflow()
    
    def _get_size_pytorch(self) -> Dict[str, float]:
        """Get PyTorch model size."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        
        total_mb = (param_size + buffer_size) / 1024 / 1024
        param_count = sum(p.numel() for p in self.model.parameters())
        
        return {
            'size_mb': total_mb,
            'param_count': param_count,
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024
        }
    
    def _get_size_tensorflow(self) -> Dict[str, float]:
        """Get TensorFlow model size."""
        param_count = self.model.count_params()
        
        # Estimate size (assuming float32)
        size_mb = (param_count * 4) / 1024 / 1024
        
        return {
            'size_mb': size_mb,
            'param_count': param_count
        }


class KnowledgeDistillation:
    """
    Knowledge distillation for model compression.
    Train a smaller student model to mimic a larger teacher.
    """
    
    def __init__(
        self,
        teacher_model,
        student_model,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        """
        Initialize knowledge distillation.
        
        Args:
            teacher_model: Trained teacher model
            student_model: Student model to train
            temperature: Softening temperature for logits
            alpha: Weight for distillation loss (1-alpha for student loss)
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(
        self,
        student_logits,
        teacher_logits,
        labels,
        framework: str
    ):
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: Ground truth labels
            framework: 'pytorch' or 'tensorflow'
            
        Returns:
            Combined loss
        """
        if framework == 'pytorch':
            import torch.nn.functional as F
            
            # Soft targets from teacher
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
            
            # Distillation loss
            distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
            distill_loss *= (self.temperature ** 2)
            
            # Student loss (standard cross-entropy)
            student_loss = F.cross_entropy(student_logits, labels)
            
            # Combined loss
            loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
            
        else:  # TensorFlow
            # Soft targets from teacher
            soft_targets = tf.nn.softmax(teacher_logits / self.temperature)
            soft_student = tf.nn.log_softmax(student_logits / self.temperature)
            
            # Distillation loss
            distill_loss = tf.reduce_mean(
                tf.reduce_sum(-soft_targets * soft_student, axis=1)
            )
            distill_loss *= (self.temperature ** 2)
            
            # Student loss
            student_loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels, student_logits, from_logits=True
            )
            student_loss = tf.reduce_mean(student_loss)
            
            # Combined loss
            loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        
        return loss


def compress_model(
    model,
    framework: str,
    method: str = 'quantization',
    **kwargs
) -> Any:
    """
    High-level function to compress a model.
    
    Args:
        model: Model to compress
        framework: 'pytorch' or 'tensorflow'
        method: Compression method ('quantization', 'pruning', 'distillation')
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Compressed model
    """
    compressor = ModelCompressor(model, framework)
    
    if method == 'quantization':
        return compressor.quantize(**kwargs)
    elif method == 'pruning':
        return compressor.prune(**kwargs)
    else:
        raise ValueError(f"Unknown compression method: {method}")


def benchmark_model(
    model,
    input_shape: Tuple[int, ...],
    framework: str,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        framework: 'pytorch' or 'tensorflow'
        num_runs: Number of inference runs
        
    Returns:
        Benchmark results
    """
    import time
    
    if framework == 'pytorch':
        import torch
        
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                times.append(time.perf_counter() - start)
    
    else:  # TensorFlow
        dummy_input = tf.random.normal(input_shape)
        
        # Warmup
        for _ in range(10):
            _ = model(dummy_input, training=False)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input, training=False)
            times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'fps': 1000 / np.mean(times)
    }
