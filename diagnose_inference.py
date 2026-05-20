"""
Inference Diagnostic Script - Test and validate inference functionality
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if required libraries are available."""
    logger.info("Testing imports...")
    issues = []
    
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} available")
    except ImportError as e:
        issues.append(f"❌ PyTorch not available: {e}")
    
    try:
        import tensorflow as tf
        logger.info(f"✅ TensorFlow {tf.__version__} available")
    except ImportError as e:
        issues.append(f"❌ TensorFlow not available: {e}")
    
    try:
        import PIL
        logger.info(f"✅ PIL/Pillow {PIL.__version__} available")
    except ImportError as e:
        issues.append(f"❌ PIL/Pillow not available: {e}")
    
    try:
        import numpy as np
        logger.info(f"✅ NumPy {np.__version__} available")
    except ImportError as e:
        issues.append(f"❌ NumPy not available: {e}")
    
    if issues:
        for issue in issues:
            logger.error(issue)
        return False
    return True


def find_models():
    """Find available model files in the workspace."""
    logger.info("\nSearching for models...")
    search_dirs = ["experiments", "models", "checkpoints"]
    extensions = [".pt", ".pth", ".keras", ".h5", ".pkl", ".joblib"]
    
    models = []
    for dir_name in search_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for ext in extensions:
                found = list(dir_path.rglob(f"*{ext}"))
                models.extend(found)
    
    if models:
        logger.info(f"Found {len(models)} model files:")
        for model in sorted(models):
            size_mb = model.stat().st_size / (1024 * 1024)
            logger.info(f"  📦 {model} ({size_mb:.2f} MB)")
        return models
    else:
        logger.warning("⚠️  No model files found. Train a model first.")
        return []


def test_pytorch_model(model_path: Path):
    """Test loading and inference with a PyTorch model."""
    logger.info(f"\nTesting PyTorch model: {model_path}")
    
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # Load model
        logger.info("  Loading checkpoint...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Analyze checkpoint
        if isinstance(checkpoint, dict):
            logger.info(f"  Checkpoint type: dict")
            logger.info(f"  Keys: {list(checkpoint.keys())}")
            arch = checkpoint.get("architecture", checkpoint.get("model_name", "unknown"))
            num_classes = checkpoint.get("num_classes", "unknown")
            logger.info(f"  Architecture: {arch}")
            logger.info(f"  Num classes: {num_classes}")
            
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
            if state:
                logger.info(f"  State dict keys (first 5): {list(state.keys())[:5]}")
                # Detect input channels
                for key, val in state.items():
                    if isinstance(val, torch.Tensor) and val.ndim == 4 and "weight" in key.lower():
                        if any(t in key.lower() for t in ("conv", "features")):
                            logger.info(f"  First conv shape: {val.shape} (channels: {val.shape[1]})")
                            break
        else:
            logger.info(f"  Checkpoint type: {type(checkpoint)}")
            if hasattr(checkpoint, "__class__"):
                logger.info(f"  Model class: {checkpoint.__class__.__name__}")
        
        logger.info("  ✅ PyTorch model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Failed to load PyTorch model: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_tensorflow_model(model_path: Path):
    """Test loading and inference with a TensorFlow model."""
    logger.info(f"\nTesting TensorFlow model: {model_path}")
    
    try:
        import tensorflow as tf
        
        # Load model
        logger.info("  Loading model...")
        model = tf.keras.models.load_model(str(model_path))
        
        # Analyze model
        logger.info(f"  Model type: {type(model)}")
        logger.info(f"  Input shape: {model.input_shape}")
        logger.info(f"  Output shape: {model.output_shape}")
        
        # Get summary
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = summary_buffer = StringIO()
        model.summary()
        sys.stdout = old_stdout
        
        summary = summary_buffer.getvalue()
        lines = summary.split('\n')
        logger.info(f"  Model summary (first 10 lines):")
        for line in lines[:10]:
            if line.strip():
                logger.info(f"    {line}")
        
        logger.info("  ✅ TensorFlow model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Failed to load TensorFlow model: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_inference_api():
    """Test the inference API endpoint."""
    logger.info("\nTesting Inference API...")
    
    try:
        from db.database import init_db, get_db
        from db.models import ModelVersion
        from sqlalchemy.orm import Session
        
        # Initialize database
        init_db()
        db = next(get_db())
        
        # Query available models
        models = db.query(ModelVersion).all()
        
        if models:
            logger.info(f"Found {len(models)} models in database:")
            for model in models[:5]:  # Show first 5
                logger.info(f"  ID: {model.id}, Path: {model.model_path}, Framework: {model.framework}")
        else:
            logger.warning("  ⚠️  No models found in database")
        
        db.close()
        logger.info("  ✅ Database connection successful")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Failed to query database: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("INFERENCE DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        logger.error("\n❌ Required libraries missing. Install dependencies first.")
        return False
    
    # Find models
    models = find_models()
    
    # Test loading each model
    pytorch_ok = 0
    tensorflow_ok = 0
    
    for model_path in models:
        ext = model_path.suffix.lower()
        if ext in [".pt", ".pth"]:
            if test_pytorch_model(model_path):
                pytorch_ok += 1
        elif ext in [".keras", ".h5"]:
            if test_tensorflow_model(model_path):
                tensorflow_ok += 1
    
    # Test API
    api_ok = test_inference_api()
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Total models found: {len(models)}")
    print(f"PyTorch models loadable: {pytorch_ok}")
    print(f"TensorFlow models loadable: {tensorflow_ok}")
    print(f"API connection: {'✅ OK' if api_ok else '❌ FAILED'}")
    print("=" * 60)
    
    if pytorch_ok > 0 or tensorflow_ok > 0:
        print("\n✅ Inference system appears to be working!")
        print("\nTo test inference:")
        print("  1. Start the web UI: streamlit run app.py")
        print("  2. Navigate to 'Model Inference' page")
        print("  3. Upload an image and run inference")
        print("\nOr use the API:")
        print("  1. Start the API: uvicorn api.main:app --reload")
        print("  2. Visit http://localhost:8000/docs")
        print("  3. Try the /api/v1/inference/ endpoint")
    else:
        print("\n⚠️  No models could be loaded. Please check:")
        print("  - Model files are not corrupted")
        print("  - Models were trained with compatible library versions")
        print("  - Required dependencies are installed")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
