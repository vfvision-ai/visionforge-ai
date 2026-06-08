"""
Data Versioning and Lineage Tracking for VisionForge v2.1

Provides comprehensive data management:
- Dataset versioning with semantic versions
- Data lineage tracking (provenance)
- Automatic change detection
- Data quality validation
- Reproducibility through snapshots
- Dataset comparison and diff
"""

import hashlib
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import pickle

logger = logging.getLogger(__name__)


class DatasetStage(Enum):
    """Dataset lifecycle stages."""
    RAW = "raw"
    PROCESSED = "processed"
    VALIDATED = "validated"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class DatasetVersion:
    """Represents a versioned dataset."""
    name: str
    version: str  # Semantic version
    stage: DatasetStage
    created_at: str
    path: str
    hash: str
    size_bytes: int
    num_samples: int
    num_classes: Optional[int]
    metadata: Dict[str, Any]
    parent_version: Optional[str] = None
    transformations: List[str] = None
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.transformations is None:
            self.transformations = []
        if self.tags is None:
            self.tags = []
        if isinstance(self.stage, str):
            self.stage = DatasetStage(self.stage)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['stage'] = self.stage.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DataLineage:
    """Tracks data lineage and provenance."""
    dataset_name: str
    dataset_version: str
    source_datasets: List[Tuple[str, str]]  # (name, version) pairs
    transformations: List[Dict[str, Any]]
    created_at: str
    created_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataLineage':
        """Create from dictionary."""
        return cls(**data)


class DataVersionControl:
    """
    Version control system for datasets.
    Similar to Git but for data.
    """
    
    def __init__(self, repository_path: Path):
        """
        Initialize data version control.
        
        Args:
            repository_path: Path to DVC repository
        """
        self.repo_path = Path(repository_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        self.versions_dir = self.repo_path / "versions"
        self.metadata_dir = self.repo_path / "metadata"
        self.lineage_dir = self.repo_path / "lineage"
        
        for dir_path in [self.versions_dir, self.metadata_dir, self.lineage_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.index_file = self.repo_path / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, List[str]]:
        """Load repository index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save repository index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _compute_dataset_hash(self, dataset_path: Path) -> str:
        """
        Compute hash for dataset.
        Uses file structure and sample of files for efficiency.
        """
        hash_md5 = hashlib.md5()
        
        if dataset_path.is_dir():
            # Hash directory structure
            for file_path in sorted(dataset_path.rglob("*")):
                if file_path.is_file():
                    # Add relative path
                    hash_md5.update(str(file_path.relative_to(dataset_path)).encode())
                    # Add file size
                    hash_md5.update(str(file_path.stat().st_size).encode())
            
            # Sample some files for content hash
            files = list(dataset_path.rglob("*"))[:100]
            for file_path in sorted(files):
                if file_path.is_file() and file_path.stat().st_size < 1024 * 1024:  # < 1MB
                    with open(file_path, 'rb') as f:
                        hash_md5.update(f.read())
        else:
            # Single file
            with open(dataset_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_dataset_size(self, dataset_path: Path) -> int:
        """Get total size of dataset in bytes."""
        if dataset_path.is_file():
            return dataset_path.stat().st_size
        
        total_size = 0
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def _count_samples(self, dataset_path: Path) -> int:
        """Estimate number of samples in dataset."""
        if not dataset_path.is_dir():
            return 1
        
        # Count image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        count = 0
        
        for file_path in dataset_path.rglob("*"):
            if file_path.suffix.lower() in image_extensions:
                count += 1
        
        return count
    
    def commit(
        self,
        name: str,
        dataset_path: Path,
        version: Optional[str] = None,
        stage: DatasetStage = DatasetStage.RAW,
        metadata: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
        transformations: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        description: str = ""
    ) -> DatasetVersion:
        """
        Commit a new dataset version.
        
        Args:
            name: Dataset name
            dataset_path: Path to dataset
            version: Version string (auto-increment if None)
            stage: Dataset stage
            metadata: Additional metadata
            parent_version: Parent version if this is derived
            transformations: Applied transformations
            tags: Tags for categorization
            description: Version description
            
        Returns:
            DatasetVersion object
        """
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Auto-generate version if not provided
        if version is None:
            version = self._next_version(name)
        
        # Compute hash and metadata
        dataset_hash = self._compute_dataset_hash(dataset_path)
        size_bytes = self._get_dataset_size(dataset_path)
        num_samples = self._count_samples(dataset_path)
        
        # Create version object
        dataset_version = DatasetVersion(
            name=name,
            version=version,
            stage=stage,
            created_at=datetime.now().isoformat(),
            path=str(dataset_path),
            hash=dataset_hash,
            size_bytes=size_bytes,
            num_samples=num_samples,
            num_classes=metadata.get('num_classes') if metadata else None,
            metadata=metadata or {},
            parent_version=parent_version,
            transformations=transformations or [],
            tags=tags or [],
            description=description
        )
        
        # Save metadata
        version_id = f"{name}_v{version}"
        metadata_file = self.metadata_dir / f"{version_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_version.to_dict(), f, indent=2)
        
        # Update index
        if name not in self.index:
            self.index[name] = []
        if version not in self.index[name]:
            self.index[name].append(version)
        self._save_index()
        
        logger.info(f"Committed dataset version: {version_id}")
        return dataset_version
    
    def _next_version(self, name: str) -> str:
        """Generate next version number."""
        if name not in self.index or not self.index[name]:
            return "1.0.0"
        
        # Get latest version and increment
        versions = self.index[name]
        latest = sorted(versions, key=lambda v: tuple(map(int, v.split('.'))))[-1]
        
        major, minor, patch = map(int, latest.split('.'))
        return f"{major}.{minor}.{patch + 1}"
    
    def get_version(self, name: str, version: Optional[str] = None) -> Optional[DatasetVersion]:
        """
        Get dataset version metadata.
        
        Args:
            name: Dataset name
            version: Version string (latest if None)
            
        Returns:
            DatasetVersion or None
        """
        if name not in self.index:
            return None
        
        if version is None:
            # Get latest version
            versions = self.index[name]
            version = sorted(versions, key=lambda v: tuple(map(int, v.split('.'))))[-1]
        
        version_id = f"{name}_v{version}"
        metadata_file = self.metadata_dir / f"{version_id}.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        return DatasetVersion.from_dict(data)
    
    def list_versions(self, name: str) -> List[DatasetVersion]:
        """List all versions of a dataset."""
        if name not in self.index:
            return []
        
        versions = []
        for version in self.index[name]:
            dataset_version = self.get_version(name, version)
            if dataset_version:
                versions.append(dataset_version)
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def diff(self, name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two dataset versions.
        
        Args:
            name: Dataset name
            version1: First version
            version2: Second version
            
        Returns:
            Diff information
        """
        v1 = self.get_version(name, version1)
        v2 = self.get_version(name, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        return {
            'version1': version1,
            'version2': version2,
            'hash_changed': v1.hash != v2.hash,
            'size_diff_bytes': v2.size_bytes - v1.size_bytes,
            'samples_diff': v2.num_samples - v1.num_samples,
            'stage_changed': v1.stage != v2.stage,
            'transformations_added': [t for t in v2.transformations if t not in v1.transformations],
            'metadata_diff': {
                k: {'v1': v1.metadata.get(k), 'v2': v2.metadata.get(k)}
                for k in set(v1.metadata.keys()) | set(v2.metadata.keys())
                if v1.metadata.get(k) != v2.metadata.get(k)
            }
        }
    
    def tag_version(self, name: str, version: str, tag: str):
        """Add tag to a version."""
        dataset_version = self.get_version(name, version)
        if not dataset_version:
            raise ValueError(f"Version {name} v{version} not found")
        
        if tag not in dataset_version.tags:
            dataset_version.tags.append(tag)
            
            # Save updated metadata
            version_id = f"{name}_v{version}"
            metadata_file = self.metadata_dir / f"{version_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(dataset_version.to_dict(), f, indent=2)
            
            logger.info(f"Tagged {version_id} with '{tag}'")
    
    def promote_version(self, name: str, version: str, new_stage: DatasetStage):
        """Promote version to new stage."""
        dataset_version = self.get_version(name, version)
        if not dataset_version:
            raise ValueError(f"Version {name} v{version} not found")
        
        dataset_version.stage = new_stage
        
        # Save updated metadata
        version_id = f"{name}_v{version}"
        metadata_file = self.metadata_dir / f"{version_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_version.to_dict(), f, indent=2)
        
        logger.info(f"Promoted {version_id} to {new_stage.value}")


class LineageTracker:
    """
    Track data lineage and provenance.
    """
    
    def __init__(self, lineage_dir: Path):
        """
        Initialize lineage tracker.
        
        Args:
            lineage_dir: Directory for lineage data
        """
        self.lineage_dir = Path(lineage_dir)
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
    
    def record_lineage(
        self,
        dataset_name: str,
        dataset_version: str,
        source_datasets: List[Tuple[str, str]],
        transformations: List[Dict[str, Any]],
        created_by: Optional[str] = None
    ) -> DataLineage:
        """
        Record data lineage.
        
        Args:
            dataset_name: Output dataset name
            dataset_version: Output dataset version
            source_datasets: List of (name, version) source datasets
            transformations: Transformations applied
            created_by: User who created this dataset
            
        Returns:
            DataLineage object
        """
        lineage = DataLineage(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            source_datasets=source_datasets,
            transformations=transformations,
            created_at=datetime.now().isoformat(),
            created_by=created_by
        )
        
        # Save lineage
        lineage_id = f"{dataset_name}_v{dataset_version}"
        lineage_file = self.lineage_dir / f"{lineage_id}_lineage.json"
        with open(lineage_file, 'w') as f:
            json.dump(lineage.to_dict(), f, indent=2)
        
        logger.info(f"Recorded lineage for {lineage_id}")
        return lineage
    
    def get_lineage(self, dataset_name: str, dataset_version: str) -> Optional[DataLineage]:
        """Get lineage for a dataset version."""
        lineage_id = f"{dataset_name}_v{dataset_version}"
        lineage_file = self.lineage_dir / f"{lineage_id}_lineage.json"
        
        if not lineage_file.exists():
            return None
        
        with open(lineage_file, 'r') as f:
            data = json.load(f)
        
        return DataLineage.from_dict(data)
    
    def trace_upstream(
        self,
        dataset_name: str,
        dataset_version: str,
        max_depth: int = 10
    ) -> List[DataLineage]:
        """
        Trace lineage upstream (find all ancestors).
        
        Args:
            dataset_name: Dataset name
            dataset_version: Dataset version
            max_depth: Maximum depth to trace
            
        Returns:
            List of lineage records
        """
        lineage_chain = []
        visited = set()
        
        def trace(name: str, version: str, depth: int):
            if depth >= max_depth or (name, version) in visited:
                return
            
            visited.add((name, version))
            lineage = self.get_lineage(name, version)
            
            if lineage:
                lineage_chain.append(lineage)
                
                # Trace sources
                for source_name, source_version in lineage.source_datasets:
                    trace(source_name, source_version, depth + 1)
        
        trace(dataset_name, dataset_version, 0)
        return lineage_chain
    
    def visualize_lineage(
        self,
        dataset_name: str,
        dataset_version: str
    ) -> str:
        """
        Generate lineage graph in DOT format.
        
        Args:
            dataset_name: Dataset name
            dataset_version: Dataset version
            
        Returns:
            DOT format graph
        """
        lineage_chain = self.trace_upstream(dataset_name, dataset_version)
        
        lines = ["digraph lineage {", "  rankdir=LR;", "  node [shape=box];"]
        
        for lineage in lineage_chain:
            target = f"{lineage.dataset_name}_v{lineage.dataset_version}"
            
            for source_name, source_version in lineage.source_datasets:
                source = f"{source_name}_v{source_version}"
                lines.append(f'  "{source}" -> "{target}";')
        
        lines.append("}")
        return '\n'.join(lines)


class DataQualityValidator:
    """
    Validate data quality before versioning.
    """
    
    def __init__(self):
        """Initialize validator."""
        self.checks: Dict[str, callable] = {}
    
    def register_check(self, name: str, check_func: callable):
        """Register a quality check."""
        self.checks[name] = check_func
    
    def validate(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Run all quality checks.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Validation results
        """
        results = {'passed': True, 'checks': {}}
        
        for name, check_func in self.checks.items():
            try:
                passed = check_func(dataset_path)
                results['checks'][name] = {
                    'passed': passed,
                    'timestamp': datetime.now().isoformat()
                }
                if not passed:
                    results['passed'] = False
            except Exception as e:
                results['checks'][name] = {
                    'passed': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results['passed'] = False
        
        return results
    
    def check_empty_dataset(self, dataset_path: Path) -> bool:
        """Check if dataset is not empty."""
        if not dataset_path.exists():
            return False
        
        if dataset_path.is_file():
            return dataset_path.stat().st_size > 0
        
        # Directory should have files
        return any(dataset_path.rglob("*"))
    
    def check_corrupt_images(self, dataset_path: Path, sample_size: int = 100) -> bool:
        """Check for corrupt images."""
        from PIL import Image
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in dataset_path.rglob("*")
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            return True  # No images to check
        
        # Sample random images
        import random
        sample = random.sample(image_files, min(sample_size, len(image_files)))
        
        corrupt_count = 0
        for img_path in sample:
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                corrupt_count += 1
        
        # Allow up to 1% corruption
        return (corrupt_count / len(sample)) < 0.01
