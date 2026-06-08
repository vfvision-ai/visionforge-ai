"""
Model Versioning and Experiment Tracking for VisionForge v2.1

Provides comprehensive model lifecycle management:
- Semantic versioning for models
- Experiment tracking with metadata
- Model lineage and provenance
- Automatic model registry
- Version comparison and rollback
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pickle

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ExperimentStatus(Enum):
    """Experiment execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelVersion:
    """Represents a versioned model."""
    name: str
    version: str  # Semantic version (e.g., "1.0.0", "2.1.3")
    framework: str  # pytorch, tensorflow, sklearn
    architecture: str
    stage: ModelStage
    created_at: str
    updated_at: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    dataset_hash: str
    model_hash: str
    parent_version: Optional[str] = None
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if isinstance(self.stage, str):
            self.stage = ModelStage(self.stage)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['stage'] = self.stage.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Experiment:
    """Represents a training experiment."""
    experiment_id: str
    name: str
    status: ExperimentStatus
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    dataset_path: str
    dataset_hash: str
    model_architecture: str
    framework: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Union[float, List[float]]]
    artifacts: Dict[str, str]  # artifact_name -> path
    tags: List[str]
    notes: str = ""
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ExperimentStatus(self.status)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Central registry for model versions.
    Tracks all models with their metadata and lineage.
    """
    
    def __init__(self, registry_path: Union[str, Path]):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.registry_path / "models"
        self.metadata_dir = self.registry_path / "metadata"
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.index_file = self.registry_path / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, List[str]]:
        """Load registry index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save registry index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register_model(
        self,
        model_version: ModelVersion,
        model_path: Path
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_version: Model version metadata
            model_path: Path to model file
            
        Returns:
            Model ID
        """
        model_id = f"{model_version.name}_v{model_version.version}"
        
        # Store model file
        model_dest = self.models_dir / f"{model_id}.pkl"
        if model_path.exists():
            import shutil
            shutil.copy2(model_path, model_dest)
        
        # Store metadata
        metadata_file = self.metadata_dir / f"{model_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2)
        
        # Update index
        if model_version.name not in self.index:
            self.index[model_version.name] = []
        self.index[model_version.name].append(model_version.version)
        self._save_index()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Retrieve model version metadata.
        
        Args:
            name: Model name
            version: Version string (if None, returns latest)
            
        Returns:
            ModelVersion or None
        """
        if name not in self.index:
            return None
        
        if version is None:
            # Get latest version
            version = self._get_latest_version(name)
        
        model_id = f"{name}_v{version}"
        metadata_file = self.metadata_dir / f"{model_id}.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        return ModelVersion.from_dict(data)
    
    def _get_latest_version(self, name: str) -> str:
        """Get latest semantic version for a model."""
        versions = self.index.get(name, [])
        if not versions:
            return None
        
        # Sort semantic versions
        def parse_version(v: str) -> tuple:
            return tuple(map(int, v.split('.')))
        
        return sorted(versions, key=parse_version)[-1]
    
    def list_models(self, stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """
        List all models in registry.
        
        Args:
            stage: Filter by stage
            
        Returns:
            List of model versions
        """
        models = []
        for name in self.index:
            for version in self.index[name]:
                model = self.get_model(name, version)
                if model and (stage is None or model.stage == stage):
                    models.append(model)
        
        return models
    
    def promote_model(self, name: str, version: str, new_stage: ModelStage):
        """
        Promote model to new stage.
        
        Args:
            name: Model name
            version: Model version
            new_stage: Target stage
        """
        model = self.get_model(name, version)
        if not model:
            raise ValueError(f"Model {name} v{version} not found")
        
        model.stage = new_stage
        model.updated_at = datetime.now().isoformat()
        
        # Save updated metadata
        model_id = f"{name}_v{version}"
        metadata_file = self.metadata_dir / f"{model_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(model.to_dict(), f, indent=2)
        
        logger.info(f"Promoted {model_id} to {new_stage.value}")
    
    def get_lineage(self, name: str, version: str) -> List[ModelVersion]:
        """
        Get model lineage (ancestry chain).
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            List of ancestor models
        """
        lineage = []
        current = self.get_model(name, version)
        
        while current and current.parent_version:
            lineage.append(current)
            current = self.get_model(name, current.parent_version)
        
        if current:
            lineage.append(current)
        
        return lineage
    
    def compare_versions(
        self,
        name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            name: Model name
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        model1 = self.get_model(name, version1)
        model2 = self.get_model(name, version2)
        
        if not model1 or not model2:
            raise ValueError("One or both models not found")
        
        return {
            'version1': version1,
            'version2': version2,
            'metrics_diff': {
                k: model2.metrics.get(k, 0) - model1.metrics.get(k, 0)
                for k in set(model1.metrics.keys()) | set(model2.metrics.keys())
            },
            'hyperparameters_diff': {
                k: {
                    'v1': model1.hyperparameters.get(k),
                    'v2': model2.hyperparameters.get(k)
                }
                for k in set(model1.hyperparameters.keys()) | set(model2.hyperparameters.keys())
                if model1.hyperparameters.get(k) != model2.hyperparameters.get(k)
            }
        }


class ExperimentTracker:
    """
    Track training experiments with full metadata.
    """
    
    def __init__(self, tracking_dir: Union[str, Path]):
        """
        Initialize experiment tracker.
        
        Args:
            tracking_dir: Directory for tracking data
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments_file = self.tracking_dir / "experiments.json"
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> Dict[str, Experiment]:
        """Load experiments from disk."""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                data = json.load(f)
            return {k: Experiment.from_dict(v) for k, v in data.items()}
        return {}
    
    def _save_experiments(self):
        """Save experiments to disk."""
        data = {k: v.to_dict() for k, v in self.experiments.items()}
        with open(self.experiments_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_experiment(
        self,
        name: str,
        dataset_path: str,
        model_architecture: str,
        framework: str,
        hyperparameters: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create new experiment.
        
        Args:
            name: Experiment name
            dataset_path: Path to dataset
            model_architecture: Model architecture name
            framework: Framework (pytorch/tensorflow/sklearn)
            hyperparameters: Training hyperparameters
            tags: Optional tags
            
        Returns:
            Experiment ID
        """
        experiment_id = self._generate_experiment_id(name)
        dataset_hash = self._hash_dataset(dataset_path)
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            status=ExperimentStatus.PENDING,
            created_at=datetime.now().isoformat(),
            started_at=None,
            completed_at=None,
            dataset_path=dataset_path,
            dataset_hash=dataset_hash,
            model_architecture=model_architecture,
            framework=framework,
            hyperparameters=hyperparameters,
            metrics={},
            artifacts={},
            tags=tags or []
        )
        
        self.experiments[experiment_id] = experiment
        self._save_experiments()
        
        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{timestamp}"
    
    def _hash_dataset(self, dataset_path: str) -> str:
        """Generate hash for dataset."""
        path = Path(dataset_path)
        if not path.exists():
            return "unknown"
        
        # Hash directory structure and a sample of files
        hash_md5 = hashlib.md5()
        
        if path.is_dir():
            for file_path in sorted(path.rglob("*"))[:100]:  # Sample first 100 files
                if file_path.is_file():
                    hash_md5.update(str(file_path.relative_to(path)).encode())
        else:
            hash_md5.update(path.name.encode())
        
        return hash_md5.hexdigest()[:16]
    
    def start_experiment(self, experiment_id: str):
        """Mark experiment as started."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        self.experiments[experiment_id].status = ExperimentStatus.RUNNING
        self.experiments[experiment_id].started_at = datetime.now().isoformat()
        self._save_experiments()
    
    def complete_experiment(
        self,
        experiment_id: str,
        metrics: Dict[str, Union[float, List[float]]],
        artifacts: Optional[Dict[str, str]] = None
    ):
        """
        Mark experiment as completed.
        
        Args:
            experiment_id: Experiment ID
            metrics: Final metrics
            artifacts: Paths to generated artifacts
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.now().isoformat()
        exp.metrics = metrics
        if artifacts:
            exp.artifacts.update(artifacts)
        
        self._save_experiments()
        logger.info(f"Completed experiment: {experiment_id}")
    
    def fail_experiment(self, experiment_id: str, error_message: str):
        """Mark experiment as failed."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.FAILED
        exp.completed_at = datetime.now().isoformat()
        exp.error_message = error_message
        
        self._save_experiments()
        logger.error(f"Failed experiment {experiment_id}: {error_message}")
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for an experiment.
        
        Args:
            experiment_id: Experiment ID
            metrics: Metrics to log
            step: Training step/epoch
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        
        for key, value in metrics.items():
            if step is not None:
                # Store as time series
                if key not in exp.metrics:
                    exp.metrics[key] = []
                exp.metrics[key].append({'step': step, 'value': value})
            else:
                # Store single value
                exp.metrics[key] = value
        
        self._save_experiments()
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.
        
        Args:
            status: Filter by status
            tags: Filter by tags (matches any)
            
        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        if tags:
            experiments = [e for e in experiments if any(t in e.tags for t in tags)]
        
        return sorted(experiments, key=lambda x: x.created_at, reverse=True)
    
    def get_best_experiment(self, metric: str, higher_is_better: bool = True) -> Optional[Experiment]:
        """
        Find best experiment based on a metric.
        
        Args:
            metric: Metric name
            higher_is_better: Whether higher values are better
            
        Returns:
            Best experiment
        """
        completed = [e for e in self.experiments.values() 
                    if e.status == ExperimentStatus.COMPLETED and metric in e.metrics]
        
        if not completed:
            return None
        
        return max(completed, key=lambda e: e.metrics[metric] if higher_is_better else -e.metrics[metric])
