"""
Dataset Analyzer - Automatically analyzes datasets and extracts metadata.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import cv2
import numpy as np
import pandas as pd


@dataclass
class DatasetInfo:
    """Dataset analysis results."""
    task_type: str
    num_classes: int
    num_samples: int
    class_names: List[str]
    class_distribution: Dict[str, int]
    image_size: Tuple[int, int]
    image_stats: Dict[str, float]
    has_annotations: bool
    annotation_format: Optional[str]
    recommended_batch_size: int
    estimated_training_time: float
    dataset_path: Optional[str] = None  # Add dataset path to make it robust
    channels: Optional[int] = None  # Number of channels for dynamic input size handling
    
    # HuggingFace dataset specific fields
    is_hf_dataset: bool = False  # Flag to identify HuggingFace datasets
    hf_dataset_name: Optional[str] = None  # HuggingFace dataset name
    hf_subset: Optional[str] = None  # HuggingFace dataset subset
    hf_features: Optional[List[str]] = None  # HuggingFace dataset features
    hf_description: Optional[str] = None  # HuggingFace dataset description
    
    # Builtin dataset specific fields
    is_builtin: bool = False  # Flag to identify builtin datasets
    builtin_dataset_name: Optional[str] = None  # Builtin dataset name
    builtin_tf_name: Optional[str] = None  # TensorFlow builtin dataset name


class DatasetAnalyzer:
    """
    Analyzes datasets to extract metadata and characteristics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def analyze(self, dataset_path: str, task_type: str = "auto") -> DatasetInfo:
        """
        Analyze dataset and extract comprehensive information.
        
        Args:
            dataset_path: Path to dataset directory
            task_type: Task type ('classification', 'detection', 'segmentation', 'auto')
            
        Returns:
            DatasetInfo object with analysis results
        """
        self.logger.info(f"Analyzing dataset at: {dataset_path}")
        
        # Handle special dataset types
        if dataset_path == "dummy":
            return self._create_dummy_dataset_info()
        
        # Check if it's a HuggingFace dataset name
        if self._is_huggingface_dataset(dataset_path):
            return self._analyze_huggingface_dataset(dataset_path, task_type)
        
        dataset_path = Path(dataset_path)
        
        # Check if it's a CSV file directly
        if dataset_path.is_file() and dataset_path.suffix.lower() == '.csv':
            return self._analyze_csv_dataset(dataset_path, task_type)
        
        # For directory-based datasets, discover images and structure
        images = self._discover_images(dataset_path)
        structure = self._analyze_structure(dataset_path)
        
        # Check for CSV files in the directory structure
        csv_files = list(dataset_path.glob("*.csv")) + list(dataset_path.glob("**/*.csv"))
        if not images and csv_files:
            # If no images but CSV files found, analyze as CSV dataset
            return self._analyze_csv_dataset(csv_files[0], task_type)
        
        if not images:
            raise ValueError("No images found in dataset")
            
        # Detect task type if auto
        if task_type == "auto":
            task_type = self._detect_task_type(dataset_path, structure, images)
        
        # Analyze based on task type
        if task_type == "classification":
            return self._analyze_classification_dataset(dataset_path, images, structure)
        elif task_type == "detection":
            return self._analyze_detection_dataset(dataset_path, images, structure)
        elif task_type == "segmentation":
            return self._analyze_segmentation_dataset(dataset_path, images, structure)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _create_dummy_dataset_info(self) -> DatasetInfo:
        """Create DatasetInfo for dummy dataset."""
        return DatasetInfo(
            task_type="classification",
            num_classes=10,
            num_samples=1000,
            class_names=[str(i) for i in range(10)],
            class_distribution={str(i): 100 for i in range(10)},
            image_size=(224, 224),
            image_stats={'mean': 0.5, 'std': 0.25},
            has_annotations=False,
            annotation_format=None,
            recommended_batch_size=32,
            estimated_training_time=0.5,
            channels=3,
            dataset_path="dummy"
        )
    
    def _is_huggingface_dataset(self, dataset_name: str) -> bool:
        """Check if dataset name is a potential HuggingFace dataset."""
        # Verified HuggingFace dataset patterns
        # NOTE: scene_parse_150 removed - it's a SEGMENTATION dataset, not suitable for classification
        hf_patterns = [
            'cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'imagenet', 'coco',
            'cats_vs_dogs', 'food101', 'keremberke/indoor-scenes-classification'
        ]
        return any(pattern in dataset_name.lower() for pattern in hf_patterns)
    
    def _analyze_huggingface_dataset(self, dataset_name: str, task_type: str = "auto") -> DatasetInfo:
        """Analyze HuggingFace dataset with VERIFIED information."""
        
        # VERIFIED dataset configurations based on actual HuggingFace Hub data
        verified_datasets = {
            'cats_vs_dogs': {
                'task_type': 'classification',
                'num_classes': 2,
                'num_samples': 23410,
                'class_names': ['cat', 'dog'],
                'image_size': (224, 224),  # Variable resolution, using common size
                'channels': 3,
                'batch_size': 32,
                'training_time': 1.0
            },
            'cifar10': {
                'task_type': 'classification', 
                'num_classes': 10,
                'num_samples': 60000,
                'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                'image_size': (32, 32),
                'channels': 3,
                'batch_size': 32,
                'training_time': 2.0
            },
            'fashion_mnist': {
                'task_type': 'classification',
                'num_classes': 10, 
                'num_samples': 70000,
                'class_names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
                'image_size': (28, 28),
                'channels': 1,
                'batch_size': 32,
                'training_time': 1.5
            },
            'food101': {
                'task_type': 'classification',
                'num_classes': 101,
                'num_samples': 101000,  # 75750 train + 25250 validation
                'class_names': [f'food_class_{i}' for i in range(101)],  # Simplified for now
                'image_size': (224, 224),  # Variable resolution, using common size
                'channels': 3,
                'batch_size': 16,  # Smaller batch due to large images
                'training_time': 4.0
            },
            # NOTE: scene_parse_150 REMOVED - It's a SEGMENTATION dataset, not suitable for classification training
            # See SCENE_PARSE_150_ANALYSIS.txt for details
            'keremberke/indoor-scenes-classification': {
                'task_type': 'classification',
                'num_classes': 67,
                'num_samples': 15620,  # Approximate total samples
                'class_names': [f'indoor_scene_{i}' for i in range(67)],
                'image_size': (224, 224),  # Variable resolution, using common size
                'channels': 3,
                'batch_size': 32,
                'training_time': 2.5
            }
        }
        
        # Get verified config or use fallback
        if dataset_name in verified_datasets:
            config = verified_datasets[dataset_name]
        else:
            # Fallback for unknown datasets
            config = {
                'task_type': 'classification',
                'num_classes': 10,
                'num_samples': 10000,
                'class_names': [f'class_{i}' for i in range(10)],
                'image_size': (224, 224),
                'channels': 3,
                'batch_size': 32,
                'training_time': 2.0
            }
        
        # Create class distribution 
        samples_per_class = config['num_samples'] // config['num_classes']
        class_distribution = {name: samples_per_class for name in config['class_names']}
        
        return DatasetInfo(
            task_type=config['task_type'],
            num_classes=config['num_classes'],
            num_samples=config['num_samples'],
            class_names=config['class_names'],
            class_distribution=class_distribution,
            image_size=config['image_size'],
            image_stats={'mean': 0.5, 'std': 0.25},
            has_annotations=False,
            annotation_format=None,
            recommended_batch_size=config['batch_size'],
            estimated_training_time=config['training_time'],
            channels=config['channels'],
            is_hf_dataset=True,
            hf_dataset_name=dataset_name,
            dataset_path=dataset_name
        )

    def _discover_images(self, dataset_path: Path) -> List[Path]:
        """Discover image files in dataset."""
        images = []
        for ext in self.supported_formats:
            images.extend(dataset_path.glob(f"**/*{ext}"))
        self.logger.info(f"Found {len(images)} images")
        return images
    
    def _analyze_structure(self, dataset_path: Path) -> Dict:
        """Analyze dataset directory structure."""
        return {
            'is_directory': dataset_path.is_dir(),
            'has_subdirs': any(p.is_dir() for p in dataset_path.iterdir()),
            'total_files': len(list(dataset_path.glob("**/*"))),
            'depth': len(dataset_path.parts)
        }
    
    def _detect_task_type(self, dataset_path: Path, structure: Dict, images: List[Path]) -> str:
        """Detect the task type based on dataset structure."""
        # Simple heuristics for task detection
        if structure['has_subdirs']:
            # If has subdirs, likely classification
            return "classification"
        else:
            # Check for annotation files
            annotation_files = list(dataset_path.glob("*.json")) + list(dataset_path.glob("*.xml"))
            if annotation_files:
                return "detection"
            else:
                return "classification"
    
    def _analyze_classification_dataset(self, dataset_path: Path, images: List[Path], structure: Dict) -> DatasetInfo:
        """Analyze classification dataset."""
        # Extract class information from directory structure
        if structure['has_subdirs']:
            class_dirs = [p for p in dataset_path.iterdir() if p.is_dir()]
            class_names = [d.name for d in class_dirs]
            class_distribution = {}
            
            for class_dir in class_dirs:
                class_images = []
                for ext in self.supported_formats:
                    class_images.extend(class_dir.glob(f"*{ext}"))
                class_distribution[class_dir.name] = len(class_images)
        else:
            # Single directory, try to infer from filenames
            class_names = ["class_0"]  # Default single class
            class_distribution = {"class_0": len(images)}
        
        # Analyze sample images
        sample_size, channels = self._analyze_sample_images(images[:10])
        
        return DatasetInfo(
            task_type="classification",
            num_classes=len(class_names),
            num_samples=len(images),
            class_names=class_names,
            class_distribution=class_distribution,
            image_size=sample_size,
            image_stats=self._calculate_image_stats(images[:50]),
            has_annotations=False,
            annotation_format=None,
            recommended_batch_size=self._recommend_batch_size(len(images)),
            estimated_training_time=self._estimate_training_time(len(images), len(class_names)),
            channels=channels,
            dataset_path=str(dataset_path)
        )
    
    def _analyze_detection_dataset(self, dataset_path: Path, images: List[Path], structure: Dict) -> DatasetInfo:
        """Analyze detection dataset — real class count from annotations when available."""
        sample_size, channels = self._analyze_sample_images(images[:10])
        num_classes = None
        class_names: List[str] = []
        annotation_format = "unknown"

        # ── 1. COCO JSON ─────────────────────────────────────────────────────
        coco_candidates = list(dataset_path.rglob("*.json"))
        for jf in coco_candidates[:5]:
            try:
                import json
                data = json.loads(jf.read_text(encoding="utf-8"))
                if "categories" in data:
                    class_names = [c["name"] for c in data["categories"]]
                    num_classes = len(class_names)
                    annotation_format = "coco"
                    break
            except Exception:
                pass

        # ── 2. YOLO labels (classes.txt / data.yaml / scan label files) ──────
        if num_classes is None:
            # Try classes.txt
            for classes_file in (dataset_path / "classes.txt",
                                  dataset_path / "obj.names",
                                  dataset_path / "data" / "obj.names"):
                if classes_file.exists():
                    lines = [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]
                    if lines:
                        class_names = lines
                        num_classes = len(class_names)
                        annotation_format = "yolo"
                        break

        if num_classes is None:
            # Try data.yaml
            for yaml_file in dataset_path.rglob("data.yaml"):
                try:
                    import yaml as _yaml
                    cfg = _yaml.safe_load(yaml_file.read_text())
                    nc = cfg.get("nc") or cfg.get("num_classes")
                    names = cfg.get("names", [])
                    if nc:
                        num_classes = int(nc)
                        class_names = list(names) if names else [f"class_{i}" for i in range(num_classes)]
                        annotation_format = "yolo"
                        break
                except Exception:
                    pass

        if num_classes is None:
            # Scan YOLO label txt files for max class id
            label_dirs = [dataset_path / "labels",
                          dataset_path / "train" / "labels",
                          dataset_path / "val"   / "labels"]
            label_files: List[Path] = []
            for d in label_dirs:
                if d.is_dir():
                    label_files.extend(list(d.glob("*.txt"))[:200])
            if label_files:
                max_id = 0
                for lf in label_files[:200]:
                    try:
                        for line in lf.read_text().splitlines():
                            parts = line.split()
                            if parts:
                                max_id = max(max_id, int(parts[0]))
                    except Exception:
                        pass
                num_classes = max_id + 1
                class_names = [f"class_{i}" for i in range(num_classes)]
                annotation_format = "yolo"

        # ── 3. Pascal VOC XML ─────────────────────────────────────────────────
        if num_classes is None:
            xml_files = list(dataset_path.rglob("*.xml"))[:200]
            if xml_files:
                try:
                    import xml.etree.ElementTree as ET
                    names_set: set = set()
                    for xf in xml_files[:200]:
                        tree = ET.parse(xf)
                        for obj in tree.iterfind(".//object/name"):
                            names_set.add(obj.text.strip())
                    if names_set:
                        class_names = sorted(names_set)
                        num_classes = len(class_names)
                        annotation_format = "voc"
                except Exception:
                    pass

        # ── 4. Fallback ───────────────────────────────────────────────────────
        if num_classes is None or num_classes < 1:
            num_classes = 10
            class_names = [f"class_{i}" for i in range(num_classes)]

        return DatasetInfo(
            task_type="detection",
            num_classes=num_classes,
            num_samples=len(images),
            class_names=class_names,
            class_distribution={n: len(images) // num_classes for n in class_names},
            image_size=sample_size,
            image_stats={'mean': 0.5, 'std': 0.25},
            has_annotations=True,
            annotation_format=annotation_format,
            recommended_batch_size=8,
            estimated_training_time=self._estimate_training_time(len(images), num_classes),
            channels=channels,
            dataset_path=str(dataset_path)
        )
    
    def _analyze_segmentation_dataset(self, dataset_path: Path, images: List[Path], structure: Dict) -> DatasetInfo:
        """Analyze segmentation dataset — detect real image size and class count from masks."""

        # Detect actual image size from sample images
        sample_size, channels = self._analyze_sample_images(images[:10])

        # Try to find mask files to determine actual class count
        num_classes = 21  # Fallback: Pascal VOC default
        class_names_detected = None

        mask_dirs_to_check = [
            dataset_path / 'masks',
            dataset_path / 'labels',
            dataset_path / 'SegmentationClass',
            dataset_path / 'annotations',
        ]
        # Also check split-specific dirs
        for split in ('train', 'val', 'test'):
            mask_dirs_to_check.append(dataset_path / split / 'masks')
            mask_dirs_to_check.append(dataset_path / split / 'labels')

        mask_files: List[Path] = []
        for mdir in mask_dirs_to_check:
            if mdir.is_dir():
                for ext in ('.png', '.bmp', '.tif', '.tiff'):
                    mask_files.extend(list(mdir.glob(f'*{ext}')))
                if mask_files:
                    break

        if mask_files:
            try:
                import numpy as np
                from PIL import Image as _PIL
                unique_values: set = set()
                for mf in mask_files[:30]:  # sample up to 30 masks
                    arr = np.array(_PIL.open(mf).convert('L'), dtype=np.uint8)
                    unique_values.update(arr.flatten().tolist())
                # Remove VOC border/ignore index (255)
                unique_values.discard(255)
                if unique_values:
                    num_classes = int(max(unique_values)) + 1
                    class_names_detected = [f"class_{i}" for i in range(num_classes)]
            except Exception:
                pass  # Keep default if PIL/numpy unavailable

        if class_names_detected is None:
            class_names_detected = [f"class_{i}" for i in range(num_classes)]

        return DatasetInfo(
            task_type="segmentation",
            num_classes=num_classes,
            num_samples=len(images),
            class_names=class_names_detected,
            class_distribution={f"class_{i}": len(images) // max(num_classes, 1) for i in range(num_classes)},
            image_size=sample_size,
            image_stats=self._calculate_image_stats(images[:50]),
            has_annotations=len(mask_files) > 0,
            annotation_format="mask",
            recommended_batch_size=4 if sample_size[0] >= 512 else 8,
            estimated_training_time=self._estimate_training_time(len(images), num_classes),
            channels=channels,
            dataset_path=str(dataset_path)
        )
    
    def _analyze_csv_dataset(self, csv_path: Path, task_type: str) -> DatasetInfo:
        """Analyze CSV dataset."""
        df = pd.read_csv(csv_path)
        
        return DatasetInfo(
            task_type="classification",
            num_classes=df.iloc[:, -1].nunique() if len(df.columns) > 1 else 2,
            num_samples=len(df),
            class_names=[str(i) for i in range(df.iloc[:, -1].nunique() if len(df.columns) > 1 else 2)],
            class_distribution=dict(df.iloc[:, -1].value_counts()) if len(df.columns) > 1 else {"0": len(df)//2, "1": len(df)//2},
            image_size=(224, 224),
            image_stats={'mean': 0.5, 'std': 0.25},
            has_annotations=False,
            annotation_format=None,
            recommended_batch_size=32,
            estimated_training_time=1.0,
            channels=3,
            dataset_path=str(csv_path)
        )
    
    def _analyze_sample_images(self, images: List[Path]) -> Tuple[Tuple[int, int], int]:
        """Analyze sample images to get size and channels."""
        if not images:
            return (224, 224), 3
        
        try:
            sample_img = cv2.imread(str(images[0]))
            if sample_img is not None:
                height, width, channels = sample_img.shape
                return (height, width), channels
            else:
                return (224, 224), 3
        except Exception:
            return (224, 224), 3
    
    def _calculate_image_stats(self, images: List[Path]) -> Dict[str, float]:
        """Calculate basic image statistics."""
        # Simplified stats calculation
        return {'mean': 0.5, 'std': 0.25}
    
    def _recommend_batch_size(self, num_samples: int) -> int:
        """Recommend batch size based on dataset size."""
        if num_samples < 100:
            return 8
        elif num_samples < 1000:
            return 16
        else:
            return 32
    
    def _estimate_training_time(self, num_samples: int, num_classes: int) -> float:
        """Estimate training time in hours."""
        # Very rough estimation based on dataset size and complexity
        base_time = (num_samples / 1000) * (num_classes / 10) * 0.5  # hours
        return max(0.5, min(base_time, 24.0))  # Clamp between 0.5 and 24 hours