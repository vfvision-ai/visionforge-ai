"""
Shared utilities used across all UI page modules.
Import this in every ui/*.py file.
"""

import streamlit as st
import os
import sys
import json
import logging
import csv
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Streamlit rerun shim ─────────────────────────────────────────────────────
def st_rerun():
    """Portable st.rerun() across Streamlit versions."""
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass


# ── Dataset class name mappings ──────────────────────────────────────────────
def get_dataset_class_names(dataset_name: str) -> Optional[List[str]]:
    """Return human-readable class names for well-known datasets, else None."""
    mappings = {
        "CIFAR-10": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        "CIFAR-100": [
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
            "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
            "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
            "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
            "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
            "flatfish", "forest", "fox", "girl", "hamster", "house",
            "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
            "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
            "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter",
            "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate",
            "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road",
            "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
            "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
            "sunflower", "sweet_pepper", "table", "tank", "telephone",
            "television", "tiger", "tractor", "train", "trout", "tulip",
            "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
        ],
        "Fashion-MNIST": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "MNIST": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    }
    return mappings.get(dataset_name)


# ── Test-sample helpers ───────────────────────────────────────────────────────
def save_test_samples_for_evaluation(
    dataset_info,
    output_dir: str,
    num_samples: int = 50,
    image_format: str = "png",
) -> None:
    """Save a random sample of test images + labels CSV for post-training evaluation."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required: pip install opencv-python")

    test_dir = os.path.join(output_dir, "test_samples")
    os.makedirs(test_dir, exist_ok=True)
    csv_path = os.path.join(test_dir, "labels.csv")

    logger.info("Saving %d test samples to %s", num_samples, test_dir)

    if getattr(dataset_info, "is_hf_dataset", False):
        _save_hf_test_samples(dataset_info, test_dir, csv_path, num_samples, image_format)
    elif getattr(dataset_info, "is_builtin", False):
        _save_builtin_test_samples(dataset_info, test_dir, csv_path, num_samples, image_format)
    elif getattr(dataset_info, "dataset_path", None):
        _save_directory_test_samples(dataset_info, test_dir, csv_path, num_samples, image_format)
    else:
        raise ValueError("Unknown dataset type — cannot save test samples.")

    logger.info("Test samples saved. Labels CSV: %s", csv_path)


def _save_hf_test_samples(dataset_info, test_dir, csv_path, num_samples, image_format="png"):
    from datasets import load_dataset
    from PIL import Image
    import cv2
    import time

    dataset_name = dataset_info.hf_dataset_name
    subset = getattr(dataset_info, "hf_subset", None)
    if subset == "None":
        subset = None

    # Temporarily hide local dir with same name to avoid HF loading it as local dataset
    local_dir = os.path.join(os.getcwd(), dataset_name)
    temp_renamed = None
    if os.path.exists(local_dir):
        temp_renamed = f"{local_dir}_tmp_{int(time.time())}"
        try:
            os.rename(local_dir, temp_renamed)
        except Exception:
            temp_renamed = None

    try:
        ds_dict = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
        available = list(ds_dict.keys())
        chosen = next((s for s in ["test", "validation", "val", "train"] if s in available), available[0])
        dataset = load_dataset(dataset_name, subset, split=chosen) if subset else load_dataset(dataset_name, split=chosen)
    finally:
        if temp_renamed and os.path.exists(temp_renamed):
            try:
                os.rename(temp_renamed, local_dir)
            except Exception:
                pass

    class_names = getattr(dataset_info, "class_names", None)
    features = list(dataset[0].keys()) if dataset else []

    label_field = next(
        (f for f in ["label", "labels", "scene_category", "category", "class", "target"] if f in features),
        next((k for k in features if any(w in k.lower() for w in ["label", "category", "class", "target"])), None),
    )
    image_field = next(
        (f for f in ["image", "img"] if f in features),
        next((k for k in features if "image" in k.lower() or "img" in k.lower()), None),
    )
    if not label_field or not image_field:
        raise ValueError(f"Cannot find label/image fields in {features}")

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    ext = "jpg" if image_format.lower() == "jpg" else "png"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "label", "label_name"])
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            img = sample[image_field]
            label = sample[label_field]
            if isinstance(img, Image.Image):
                arr = np.array(img)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                arr = np.array(img)
            fname = f"test_sample_{i:04d}.{ext}"
            cv2.imwrite(os.path.join(test_dir, fname), arr)
            lname = class_names[label] if class_names and label < len(class_names) else f"class_{label}"
            w.writerow([fname, label, lname])


def _save_builtin_test_samples(dataset_info, test_dir, csv_path, num_samples, image_format="png"):
    import tensorflow as tf
    import cv2

    name = (getattr(dataset_info, "builtin_dataset_name", "") or
            getattr(dataset_info, "builtin_tf_name", "") or "").lower()

    dataset_loaders = {
        ("cifar10", "cifar-10", "cifar_10"): (
            lambda: tf.keras.datasets.cifar10.load_data(),
            ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        ),
        ("cifar100", "cifar-100", "cifar_100"): (
            lambda: tf.keras.datasets.cifar100.load_data(),
            [f"class_{i}" for i in range(100)],
        ),
        ("mnist",): (
            lambda: tf.keras.datasets.mnist.load_data(),
            [str(i) for i in range(10)],
        ),
        ("fashion_mnist", "fashion-mnist"): (
            lambda: tf.keras.datasets.fashion_mnist.load_data(),
            ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
             "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        ),
    }

    loader, class_names = None, []
    for keys, (fn, names) in dataset_loaders.items():
        if name in keys:
            loader, class_names = fn, names
            break
    if loader is None:
        raise ValueError(f"Unsupported built-in dataset: {name}")

    (_, _), (x_test, y_test) = loader()
    indices = random.sample(range(len(x_test)), min(num_samples, len(x_test)))
    ext = "jpg" if image_format.lower() == "jpg" else "png"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "label", "label_name"])
        for i, idx in enumerate(indices):
            img = x_test[idx]
            label = int(y_test[idx].flat[0]) if hasattr(y_test[idx], "flat") else int(y_test[idx])
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            fname = f"test_sample_{i:04d}.{ext}"
            cv2.imwrite(os.path.join(test_dir, fname), img)
            lname = class_names[label] if label < len(class_names) else f"class_{label}"
            w.writerow([fname, label, lname])


def _save_directory_test_samples(dataset_info, test_dir, csv_path, num_samples, image_format="png"):
    import cv2

    dataset_path = dataset_info.dataset_path
    candidate_dirs = ["test", "testing", "val", "validation", "train"]
    src_dir = next(
        (os.path.join(dataset_path, d) for d in candidate_dirs if os.path.exists(os.path.join(dataset_path, d))),
        None,
    )
    if not src_dir:
        raise ValueError("Could not find test/train directory in dataset path.")

    image_files = []
    for cls_dir in os.listdir(src_dir):
        cls_path = os.path.join(src_dir, cls_dir)
        if os.path.isdir(cls_path):
            label_idx = len(image_files)  # simple incremental index
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    image_files.append((os.path.join(cls_path, fname), label_idx, cls_dir))

    if not image_files:
        raise ValueError("No image files found.")

    samples = random.sample(image_files, min(num_samples, len(image_files)))
    ext = "jpg" if image_format.lower() == "jpg" else "png"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "label", "label_name"])
        for i, (src, label, lname) in enumerate(samples):
            img = cv2.imread(src)
            if img is None:
                continue
            fname = f"test_sample_{i:04d}.{ext}"
            cv2.imwrite(os.path.join(test_dir, fname), img)
            w.writerow([fname, label, lname])
