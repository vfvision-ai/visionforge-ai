"""Quick script to inspect checkpoint normalization"""
import sys
from pathlib import Path

# Find the most recent model checkpoint
search_dirs = ["experiments", "models", "checkpoints"]
extensions = [".pt", ".pth"]

models = []
for dir_name in search_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        for ext in extensions:
            found = list(dir_path.rglob(f"*{ext}"))
            models.extend(found)

if not models:
    print("❌ No PyTorch models found!")
    sys.exit(1)

# Get the most recent one
latest_model = max(models, key=lambda f: f.stat().st_mtime)
print(f"📦 Inspecting: {latest_model}")
print(f"   Size: {latest_model.stat().st_size / (1024*1024):.2f} MB")
print(f"   Modified: {latest_model.stat().st_mtime}")
print()

try:
    import torch
    checkpoint = torch.load(latest_model, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict):
        print("✅ Checkpoint is a dictionary")
        print(f"   Keys: {list(checkpoint.keys())}")
        print()
        
        # Check normalization metadata
        norm_mean = checkpoint.get("normalization_mean")
        norm_std = checkpoint.get("normalization_std")
        
        print("🔍 Normalization Metadata:")
        print(f"   normalization_mean: {norm_mean}")
        print(f"   normalization_std: {norm_std}")
        print()
        
        if norm_mean is None and norm_std is None:
            print("⚠️  NO NORMALIZATION METADATA FOUND!")
            print("   This means the model was trained with Standard (0-1)")
            print("   But inference might be expecting something different!")
        elif norm_mean == [0.5] and norm_std == [0.5]:
            print("✅ Z-Score normalization for grayscale: [-1, 1] range")
        elif norm_mean == [0.5, 0.5, 0.5] and norm_std == [0.5, 0.5, 0.5]:
            print("✅ Z-Score normalization for RGB: [-1, 1] range")
        elif norm_mean == [0.485, 0.456, 0.406]:
            print("✅ ImageNet normalization (ResNet/MobileNet)")
        else:
            print(f"✅ Custom normalization")
        
        print()
        print("📊 Other Metadata:")
        print(f"   architecture: {checkpoint.get('architecture')}")
        print(f"   num_classes: {checkpoint.get('num_classes')}")
        print(f"   image_size: {checkpoint.get('image_size')}")
        print(f"   channels: {checkpoint.get('channels')}")
        print(f"   dataset_name: {checkpoint.get('dataset_name')}")
        print(f"   best_accuracy: {checkpoint.get('best_accuracy')}")
        
    else:
        print("⚠️  Checkpoint is a raw model object, no metadata available")
        
except ImportError:
    print("❌ PyTorch not available. Cannot inspect checkpoint.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
