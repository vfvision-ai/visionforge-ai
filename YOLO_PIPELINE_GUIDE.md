# YOLO Training Pipeline - User Guide

## Overview
The platform now supports the **official Ultralytics YOLO pipeline** for object detection tasks, providing state-of-the-art performance with an easy-to-use interface.

---

## Features

✅ **Standard Ultralytics API**: Uses the official YOLO training pipeline  
✅ **Multiple Variants**: Choose from YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x  
✅ **Smart Auto-Selection**: Automatic variant selection based on dataset size  
✅ **Full Specifications**: See exact parameters, speed, and mAP for each variant  
✅ **Framework-Aware**: Only available in PyTorch (proper validation for other frameworks)

---

## Supported YOLO Variants

| Variant | Parameters | Speed | mAP@0.5 | Best For |
|---------|-----------|-------|---------|----------|
| **YOLOv8n** | 3.2M | ⚡⚡⚡⚡⚡ | 37.3% | Edge devices, real-time on CPU |
| **YOLOv8s** | 11.2M | ⚡⚡⚡⚡ | 44.9% | Best balance, production use |
| **YOLOv8m** | 25.9M | ⚡⚡⚡ | 50.2% | High accuracy, GPU recommended |
| **YOLOv8l** | 43.7M | ⚡⚡ | 52.9% | Maximum accuracy, large datasets |
| **YOLOv8x** | 68.2M | ⚡ | 53.9% | Research, maximum accuracy |

---

## How to Use

### Step 1: Select Framework
- Choose **PyTorch** framework (YOLO only works with PyTorch)
- TensorFlow and Scikit-learn will show a clear error for detection tasks

### Step 2: Select YOLO Model
In the Model Selection page:
1. Select any YOLO model from the dropdown
2. A **YOLO Configuration** section will appear
3. Choose your preferred variant from the dropdown
4. See specifications (parameters, speed, mAP) for the selected variant
5. Enable "✅ Use Ultralytics YOLO Pipeline (Recommended)"

### Step 3: Configure Training
- Set epochs, batch size, learning rate as usual
- YOLO will use standard ultralytics training API
- All metrics are automatically tracked (mAP50, mAP50-95, precision, recall)

### Step 4: Start Training
- Click "🚀 Start Training"
- Training uses the official ultralytics pipeline
- Results include comprehensive metrics and plots

---

## Dataset Format

YOLO supports multiple annotation formats:

### **YOLO Format** (Recommended)
```
dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/  (.txt files)
  │   └── val/    (.txt files)
  └── data.yaml
```

### **COCO Format**
```
dataset/
  ├── images/
  └── annotations/
      ├── instances_train.json
      └── instances_val.json
```

### **Pascal VOC Format**
```
dataset/
  ├── JPEGImages/
  └── Annotations/  (.xml files)
```

---

## Auto-Selection Logic

The system automatically suggests the best YOLO variant based on your dataset:

- **< 500 images**: YOLOv8n (prevents overfitting, fast training)
- **500-2000 images**: YOLOv8s (best balance)
- **2000-5000 images**: YOLOv8m (better accuracy)
- **5000-10000 images**: YOLOv8l (leverage larger dataset)
- **> 10000 images**: YOLOv8l (maximum accuracy)

---

## Training Outputs

After training, you'll get:
- ✅ Best model weights (`best.pt`)
- ✅ Training metrics (mAP50, mAP50-95, precision, recall)
- ✅ Training history CSV
- ✅ Confusion matrix and plots
- ✅ Val/Train loss curves

---

## Requirements

```bash
pip install ultralytics==8.0.196
```

---

## Technical Implementation

### Files Modified/Created:
1. **`core/yolo_trainer.py`**: New YOLO trainer using ultralytics API
2. **`ui/models.py`**: YOLO version selector UI
3. **`ui/helpers.py`**: Updated model list with YOLO variants
4. **`workers/training_tasks.py`**: Updated to use YOLOTrainer for YOLO models
5. **`utils/data_factory.py`**: Detection dataset loading support

### Key Components:

**YOLOTrainer Class**:
- Uses `ultralytics.YOLO` API
- Handles data.yaml creation
- Automatic model download
- Standard ultralytics training parameters
- Proper metric extraction

**UI Integration**:
- Dropdown for variant selection
- Live specifications display
- Auto-suggestions based on dataset
- Framework validation

**Training Task Integration**:
- Detects YOLO model selection
- Routes to YOLOTrainer instead of AutoTrainer
- Passes variant and pipeline flags
- Maintains result consistency

---

## Framework Support Matrix

| Task | PyTorch | TensorFlow | Scikit-learn |
|------|---------|------------|--------------|
| Classification | ✅ | ✅ | ✅ |
| Detection (YOLO) | ✅ | ❌ | ❌ |
| Segmentation | ✅ | ❌ | ❌ |

---

## Troubleshooting

**Q: I selected YOLO but training fails**  
A: Check if ultralytics is installed: `pip install ultralytics`

**Q: Dataset not loading**  
A: Ensure your dataset is in YOLO, COCO, or VOC format with proper structure

**Q: Which YOLO version to choose?**  
A: Use the auto-suggestion, or:
  - Small datasets (< 2000): YOLOv8n or YOLOv8s
  - Medium datasets (2000-5000): YOLOv8m
  - Large datasets (> 5000): YOLOv8l

**Q: Can I use TensorFlow for YOLO?**  
A: No, YOLO training requires PyTorch. The UI will show a clear error if you try.

---

## Example Usage

```python
# The YOLOTrainer can also be used directly:
from core.yolo_trainer import YOLOTrainer
from utils.config import Config

# Create config
config = Config(...)

# Initialize trainer with variant
trainer = YOLOTrainer(config, yolo_variant="yolov8m")

# Train
results = trainer.train()

# Results include mAP50, mAP50-95, precision, recall
print(f"mAP50: {results['final_metrics']['map50']}")
```

---

## References

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **YOLOv8 Paper**: https://arxiv.org/abs/2305.09972
- **Model Zoo**: https://github.com/ultralytics/ultralytics

---

**Created**: June 8, 2026  
**Version**: 1.0  
**Status**: Production Ready ✅
