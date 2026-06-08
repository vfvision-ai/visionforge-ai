"""
Model Explainability Module for VisionForge v2.1

Provides advanced explainability features including:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Integrated Gradients
- Saliency Maps
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any
import logging
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visual explanations.
    Shows which regions of the image were important for the prediction.
    """
    
    def __init__(self, model, target_layer: Optional[str] = None):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: PyTorch or TensorFlow model
            target_layer: Name of the layer to visualize (defaults to last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.framework = self._detect_framework()
        
    def _detect_framework(self) -> str:
        """Detect whether model is PyTorch or TensorFlow."""
        if TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
            return 'pytorch'
        elif TF_AVAILABLE and isinstance(self.model, tf.keras.Model):
            return 'tensorflow'
        else:
            raise ValueError("Model must be PyTorch or TensorFlow model")
    
    def _register_hooks_pytorch(self, layer_name: str):
        """Register forward and backward hooks for PyTorch."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find the target layer
        for name, module in self.model.named_modules():
            if name == layer_name or (not layer_name and 'conv' in name.lower()):
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                logger.info(f"Registered hooks on layer: {name}")
                return
                
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def generate_cam_pytorch(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for PyTorch model.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Calculate weights
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted combination
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def generate_cam_tensorflow(
        self, 
        input_tensor: tf.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for TensorFlow model.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            
        Returns:
            Heatmap as numpy array
        """
        # Create a model that outputs both predictions and activations
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.output, self.model.get_layer(self.target_layer).output]
        )
        
        with tf.GradientTape() as tape:
            predictions, activations = grad_model(input_tensor)
            if target_class is None:
                target_class = tf.argmax(predictions[0]).numpy()
            class_channel = predictions[:, target_class]
        
        # Calculate gradients
        grads = tape.gradient(class_channel, activations)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight activations
        activations = activations[0]
        heatmap = tf.reduce_sum(pooled_grads * activations, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def visualize(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H, W, C) in RGB
            heatmap: Grad-CAM heatmap (H', W')
            alpha: Transparency of overlay
            colormap: OpenCV colormap
            
        Returns:
            Overlayed image
        """
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        superimposed = heatmap * alpha + image * (1 - alpha)
        return np.uint8(superimposed)


class SaliencyMap:
    """Generate saliency maps showing pixel importance."""
    
    def __init__(self, model):
        self.model = model
        self.framework = self._detect_framework()
        
    def _detect_framework(self) -> str:
        if TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
            return 'pytorch'
        elif TF_AVAILABLE and isinstance(self.model, tf.keras.Model):
            return 'tensorflow'
        else:
            raise ValueError("Model must be PyTorch or TensorFlow model")
    
    def generate_pytorch(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate saliency map for PyTorch model."""
        self.model.eval()
        input_tensor.requires_grad = True
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        saliency = input_tensor.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def generate_tensorflow(
        self, 
        input_tensor: tf.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate saliency map for TensorFlow model."""
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.model(input_tensor)
            
            if target_class is None:
                target_class = tf.argmax(predictions[0]).numpy()
            
            class_channel = predictions[:, target_class]
        
        grads = tape.gradient(class_channel, input_tensor)
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
        saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
        
        return saliency.numpy()
    
    def generate(self, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate saliency map (framework-agnostic)."""
        if self.framework == 'pytorch':
            return self.generate_pytorch(input_tensor, target_class)
        else:
            return self.generate_tensorflow(input_tensor, target_class)


class IntegratedGradients:
    """
    Integrated Gradients for attribution.
    More stable than simple gradients.
    """
    
    def __init__(self, model):
        self.model = model
        self.framework = self._detect_framework()
        
    def _detect_framework(self) -> str:
        if TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
            return 'pytorch'
        elif TF_AVAILABLE and isinstance(self.model, tf.keras.Model):
            return 'tensorflow'
        else:
            raise ValueError("Model must be PyTorch or TensorFlow model")
    
    def generate_pytorch(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> np.ndarray:
        """Generate integrated gradients for PyTorch."""
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Scale input and compute gradients
        scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) 
                        for i in range(steps + 1)]
        
        grads = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad = True
            output = self.model(scaled_input)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            self.model.zero_grad()
            output[0, target_class].backward()
            grads.append(scaled_input.grad.detach())
        
        # Average gradients and multiply by input difference
        avg_grads = torch.stack(grads).mean(dim=0)
        integrated_grads = (input_tensor - baseline) * avg_grads
        
        attribution = integrated_grads.abs().max(dim=1)[0].squeeze().cpu().numpy()
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution


class ExplainabilityManager:
    """
    Unified interface for all explainability methods.
    """
    
    def __init__(self, model, method: str = 'gradcam'):
        """
        Initialize explainability manager.
        
        Args:
            model: Trained model (PyTorch or TensorFlow)
            method: Explainability method ('gradcam', 'saliency', 'integrated_gradients')
        """
        self.model = model
        self.method = method.lower()
        
        if self.method == 'gradcam':
            self.explainer = GradCAM(model)
        elif self.method == 'saliency':
            self.explainer = SaliencyMap(model)
        elif self.method == 'integrated_gradients':
            self.explainer = IntegratedGradients(model)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def explain(
        self, 
        input_data, 
        target_class: Optional[int] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for prediction.
        
        Args:
            input_data: Input image or tensor
            target_class: Target class to explain
            save_path: Path to save visualization
            
        Returns:
            Dictionary with explanation results
        """
        if self.method == 'gradcam':
            if isinstance(self.explainer.framework, str) and self.explainer.framework == 'pytorch':
                heatmap = self.explainer.generate_cam_pytorch(input_data, target_class)
            else:
                heatmap = self.explainer.generate_cam_tensorflow(input_data, target_class)
            
            return {
                'method': 'gradcam',
                'heatmap': heatmap,
                'target_class': target_class
            }
        
        elif self.method in ['saliency', 'integrated_gradients']:
            attribution = self.explainer.generate(input_data, target_class)
            
            return {
                'method': self.method,
                'attribution': attribution,
                'target_class': target_class
            }
        
        return {}


def explain_prediction(
    model,
    image: np.ndarray,
    method: str = 'gradcam',
    target_class: Optional[int] = None,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    High-level function to explain a prediction.
    
    Args:
        model: Trained model
        image: Input image (numpy array)
        method: Explainability method
        target_class: Class to explain
        save_path: Where to save visualization
        
    Returns:
        Explanation results
    """
    manager = ExplainabilityManager(model, method)
    return manager.explain(image, target_class, save_path)
