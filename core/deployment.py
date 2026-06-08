"""
Model Deployment and A/B Testing Framework for VisionForge v2.2

Provides advanced deployment capabilities:
- Model serving with multiple endpoints
- A/B testing and canary deployments
- Traffic routing and load balancing
- Model performance tracking in production
- Automated rollback on performance degradation
- Multi-model ensemble serving
"""

import logging
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    CANARY = "canary"
    SHADOW = "shadow"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    DEPRECATED = "deprecated"


class TrafficStrategy(Enum):
    """Traffic routing strategy."""
    ALL_AT_ONCE = "all_at_once"  # Replace immediately
    CANARY = "canary"  # Gradual rollout
    BLUE_GREEN = "blue_green"  # Switch between two versions
    SHADOW = "shadow"  # Shadow traffic without affecting responses
    A_B_TEST = "ab_test"  # Fixed percentage split


@dataclass
class ModelEndpoint:
    """Represents a deployed model endpoint."""
    endpoint_id: str
    model_name: str
    model_version: str
    deployment_time: str
    status: DeploymentStatus
    traffic_percentage: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'endpoint_id': self.endpoint_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'deployment_time': self.deployment_time,
            'status': self.status.value,
            'traffic_percentage': self.traffic_percentage,
            'metadata': self.metadata,
            'metrics': self.metrics
        }


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    name: str
    variants: List[Dict[str, Any]]  # List of {model_version, traffic_percentage}
    start_time: str
    end_time: Optional[str] = None
    success_metrics: List[str] = field(default_factory=list)
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'variants': self.variants,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'success_metrics': self.success_metrics,
            'minimum_sample_size': self.minimum_sample_size,
            'confidence_level': self.confidence_level
        }


class ModelDeploymentManager:
    """
    Manage model deployments with various strategies.
    """
    
    def __init__(self, deployment_dir: Path):
        """
        Initialize deployment manager.
        
        Args:
            deployment_dir: Directory for deployment metadata
        """
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        self.endpoints_file = self.deployment_dir / "endpoints.json"
        self.endpoints = self._load_endpoints()
        
        self.active_deployments: Dict[str, ModelEndpoint] = {}
        self._load_active_deployments()
    
    def _load_endpoints(self) -> Dict[str, ModelEndpoint]:
        """Load endpoint configurations."""
        if self.endpoints_file.exists():
            with open(self.endpoints_file, 'r') as f:
                data = json.load(f)
            return {k: ModelEndpoint(**v) for k, v in data.items()}
        return {}
    
    def _save_endpoints(self):
        """Save endpoint configurations."""
        data = {k: v.to_dict() for k, v in self.endpoints.items()}
        for k in data:
            if isinstance(data[k]['status'], DeploymentStatus):
                data[k]['status'] = data[k]['status'].value
        
        with open(self.endpoints_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_active_deployments(self):
        """Load active deployments."""
        for endpoint_id, endpoint in self.endpoints.items():
            if endpoint.status == DeploymentStatus.ACTIVE:
                self.active_deployments[endpoint_id] = endpoint
    
    def deploy_model(
        self,
        model_name: str,
        model_version: str,
        strategy: TrafficStrategy = TrafficStrategy.ALL_AT_ONCE,
        traffic_percentage: float = 100.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Deploy a model with specified strategy.
        
        Args:
            model_name: Model name
            model_version: Model version
            strategy: Deployment strategy
            traffic_percentage: Initial traffic percentage
            metadata: Additional metadata
            
        Returns:
            Endpoint ID
        """
        endpoint_id = self._generate_endpoint_id(model_name, model_version)
        
        # Determine initial status based on strategy
        if strategy == TrafficStrategy.ALL_AT_ONCE:
            status = DeploymentStatus.ACTIVE
            traffic = 100.0
        elif strategy == TrafficStrategy.CANARY:
            status = DeploymentStatus.CANARY
            traffic = min(traffic_percentage, 10.0)  # Start with small percentage
        elif strategy == TrafficStrategy.SHADOW:
            status = DeploymentStatus.SHADOW
            traffic = 0.0  # No real traffic
        else:
            status = DeploymentStatus.PENDING
            traffic = traffic_percentage
        
        endpoint = ModelEndpoint(
            endpoint_id=endpoint_id,
            model_name=model_name,
            model_version=model_version,
            deployment_time=datetime.now().isoformat(),
            status=status,
            traffic_percentage=traffic,
            metadata=metadata or {}
        )
        
        self.endpoints[endpoint_id] = endpoint
        
        # Update active deployments
        if status == DeploymentStatus.ACTIVE:
            # Deactivate old deployments for this model
            for old_endpoint in list(self.active_deployments.values()):
                if old_endpoint.model_name == model_name:
                    old_endpoint.status = DeploymentStatus.DEPRECATED
            
            self.active_deployments[endpoint_id] = endpoint
        
        self._save_endpoints()
        logger.info(f"Deployed model {model_name} v{model_version} with strategy {strategy.value}")
        
        return endpoint_id
    
    def _generate_endpoint_id(self, model_name: str, model_version: str) -> str:
        """Generate unique endpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_v{model_version}_{timestamp}"
    
    def update_traffic(self, endpoint_id: str, new_percentage: float):
        """
        Update traffic percentage for an endpoint.
        
        Args:
            endpoint_id: Endpoint ID
            new_percentage: New traffic percentage
        """
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_id} not found")
        
        self.endpoints[endpoint_id].traffic_percentage = new_percentage
        
        # Update status if reaching 100%
        if new_percentage >= 100.0:
            self.endpoints[endpoint_id].status = DeploymentStatus.ACTIVE
        
        self._save_endpoints()
        logger.info(f"Updated traffic for {endpoint_id} to {new_percentage}%")
    
    def canary_rollout(
        self,
        endpoint_id: str,
        target_percentage: float = 100.0,
        increment_percentage: float = 10.0,
        check_health: Optional[Callable] = None
    ) -> bool:
        """
        Gradually increase traffic for canary deployment.
        
        Args:
            endpoint_id: Endpoint ID
            target_percentage: Target traffic percentage
            increment_percentage: Increment per step
            check_health: Function to check health before incrementing
            
        Returns:
            True if successful, False if rolled back
        """
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_id} not found")
        
        endpoint = self.endpoints[endpoint_id]
        current = endpoint.traffic_percentage
        
        while current < target_percentage:
            # Check health if provided
            if check_health and not check_health(endpoint):
                logger.warning(f"Health check failed for {endpoint_id}, rolling back")
                self.rollback(endpoint_id)
                return False
            
            # Increment traffic
            current = min(current + increment_percentage, target_percentage)
            self.update_traffic(endpoint_id, current)
            
            # Wait between increments (in production, this would be time-based)
            logger.info(f"Canary rollout: {endpoint_id} at {current}%")
        
        logger.info(f"Canary rollout completed for {endpoint_id}")
        return True
    
    def rollback(self, endpoint_id: str):
        """
        Rollback a deployment.
        
        Args:
            endpoint_id: Endpoint to rollback
        """
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_id} not found")
        
        self.endpoints[endpoint_id].status = DeploymentStatus.ROLLED_BACK
        self.endpoints[endpoint_id].traffic_percentage = 0.0
        
        if endpoint_id in self.active_deployments:
            del self.active_deployments[endpoint_id]
        
        # Reactivate previous version
        model_name = self.endpoints[endpoint_id].model_name
        previous_endpoints = [
            ep for ep in self.endpoints.values()
            if ep.model_name == model_name and ep.endpoint_id != endpoint_id
            and ep.status == DeploymentStatus.DEPRECATED
        ]
        
        if previous_endpoints:
            # Find most recent
            previous = max(previous_endpoints, key=lambda x: x.deployment_time)
            previous.status = DeploymentStatus.ACTIVE
            previous.traffic_percentage = 100.0
            self.active_deployments[previous.endpoint_id] = previous
            logger.info(f"Rolled back to {previous.endpoint_id}")
        
        self._save_endpoints()
    
    def get_endpoint_for_request(self, model_name: str) -> Optional[ModelEndpoint]:
        """
        Get endpoint to serve a request based on traffic routing.
        
        Args:
            model_name: Model name
            
        Returns:
            Selected endpoint
        """
        # Get all active endpoints for this model
        candidates = [
            ep for ep in self.active_deployments.values()
            if ep.model_name == model_name and ep.traffic_percentage > 0
        ]
        
        if not candidates:
            return None
        
        # Weighted random selection based on traffic percentage
        total_traffic = sum(ep.traffic_percentage for ep in candidates)
        rand_val = random.random() * total_traffic
        
        cumulative = 0
        for endpoint in candidates:
            cumulative += endpoint.traffic_percentage
            if rand_val <= cumulative:
                return endpoint
        
        return candidates[-1]  # Fallback
    
    def list_deployments(
        self,
        model_name: Optional[str] = None,
        status: Optional[DeploymentStatus] = None
    ) -> List[ModelEndpoint]:
        """
        List deployments with optional filtering.
        
        Args:
            model_name: Filter by model name
            status: Filter by status
            
        Returns:
            List of endpoints
        """
        endpoints = list(self.endpoints.values())
        
        if model_name:
            endpoints = [ep for ep in endpoints if ep.model_name == model_name]
        
        if status:
            endpoints = [ep for ep in endpoints if ep.status == status]
        
        return sorted(endpoints, key=lambda x: x.deployment_time, reverse=True)


class ABTestManager:
    """
    Manage A/B tests for models.
    """
    
    def __init__(self, test_dir: Path):
        """
        Initialize A/B test manager.
        
        Args:
            test_dir: Directory for test data
        """
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        self.tests_file = self.test_dir / "ab_tests.json"
        self.tests = self._load_tests()
        
        self.results: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    def _load_tests(self) -> Dict[str, ABTestConfig]:
        """Load A/B test configurations."""
        if self.tests_file.exists():
            with open(self.tests_file, 'r') as f:
                data = json.load(f)
            return {k: ABTestConfig(**v) for k, v in data.items()}
        return {}
    
    def _save_tests(self):
        """Save A/B test configurations."""
        data = {k: v.to_dict() for k, v in self.tests.items()}
        with open(self.tests_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_test(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        success_metrics: List[str],
        duration_days: int = 7,
        minimum_sample_size: int = 1000,
        confidence_level: float = 0.95
    ) -> str:
        """
        Create new A/B test.
        
        Args:
            name: Test name
            variants: List of variants with model_version and traffic_percentage
            success_metrics: Metrics to evaluate
            duration_days: Test duration in days
            minimum_sample_size: Minimum samples per variant
            confidence_level: Statistical confidence level
            
        Returns:
            Test ID
        """
        # Validate traffic percentages sum to 100
        total_traffic = sum(v.get('traffic_percentage', 0) for v in variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100%, got {total_traffic}%")
        
        test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        start_time = datetime.now()
        end_time = start_time + timedelta(days=duration_days)
        
        test_config = ABTestConfig(
            test_id=test_id,
            name=name,
            variants=variants,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            success_metrics=success_metrics,
            minimum_sample_size=minimum_sample_size,
            confidence_level=confidence_level
        )
        
        self.tests[test_id] = test_config
        self._save_tests()
        
        logger.info(f"Created A/B test: {test_id}")
        return test_id
    
    def record_result(
        self,
        test_id: str,
        variant_id: str,
        metrics: Dict[str, float]
    ):
        """
        Record result for a variant.
        
        Args:
            test_id: Test ID
            variant_id: Variant identifier
            metrics: Metrics to record
        """
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        for metric_name, value in metrics.items():
            self.results[test_id][f"{variant_id}_{metric_name}"].append(value)
    
    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """
        Analyze A/B test results.
        
        Args:
            test_id: Test ID
            
        Returns:
            Analysis results
        """
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.tests[test_id]
        results = {}
        
        # Check if minimum sample size is met
        for variant in test_config.variants:
            variant_id = variant['model_version']
            for metric in test_config.success_metrics:
                key = f"{variant_id}_{metric}"
                sample_size = len(self.results[test_id].get(key, []))
                
                if sample_size < test_config.minimum_sample_size:
                    return {
                        'status': 'insufficient_data',
                        'message': f'Need {test_config.minimum_sample_size} samples, have {sample_size}'
                    }
        
        # Perform statistical comparison
        try:
            from scipy import stats
        except ImportError:
            raise ImportError(
                "scipy is required for A/B testing. Install it with: pip install scipy>=1.9.0"
            )
        import numpy as np
        
        variant_results = {}
        for variant in test_config.variants:
            variant_id = variant['model_version']
            variant_metrics = {}
            
            for metric in test_config.success_metrics:
                key = f"{variant_id}_{metric}"
                values = self.results[test_id].get(key, [])
                
                variant_metrics[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'count': len(values),
                    'median': float(np.median(values))
                }
            
            variant_results[variant_id] = variant_metrics
        
        # Compare variants (assuming 2 variants for simplicity)
        if len(test_config.variants) == 2:
            v1_id = test_config.variants[0]['model_version']
            v2_id = test_config.variants[1]['model_version']
            
            comparisons = {}
            for metric in test_config.success_metrics:
                v1_key = f"{v1_id}_{metric}"
                v2_key = f"{v2_id}_{metric}"
                
                v1_values = self.results[test_id].get(v1_key, [])
                v2_values = self.results[test_id].get(v2_key, [])
                
                # Perform t-test
                statistic, p_value = stats.ttest_ind(v1_values, v2_values)
                
                comparisons[metric] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < (1 - test_config.confidence_level),
                    'winner': v1_id if np.mean(v1_values) > np.mean(v2_values) else v2_id
                }
            
            results['comparisons'] = comparisons
        
        results['variants'] = variant_results
        results['status'] = 'complete'
        
        # Determine overall winner
        winners = [comp['winner'] for comp in results.get('comparisons', {}).values() 
                   if comp['significant']]
        
        if winners:
            from collections import Counter
            winner_counts = Counter(winners)
            overall_winner = winner_counts.most_common(1)[0][0]
            results['overall_winner'] = overall_winner
        else:
            results['overall_winner'] = None
            results['conclusion'] = 'No statistically significant difference'
        
        return results
    
    def get_winning_variant(self, test_id: str) -> Optional[str]:
        """
        Get the winning variant ID.
        
        Args:
            test_id: Test ID
            
        Returns:
            Winning variant ID or None
        """
        analysis = self.analyze_test(test_id)
        return analysis.get('overall_winner')


class ModelEnsemble:
    """
    Manage model ensembles for serving.
    """
    
    def __init__(self):
        """Initialize model ensemble."""
        self.models: List[Any] = []
        self.weights: List[float] = []
        self.ensemble_method: str = 'average'
    
    def add_model(self, model: Any, weight: float = 1.0):
        """
        Add model to ensemble.
        
        Args:
            model: Model to add
            weight: Model weight
        """
        self.models.append(model)
        self.weights.append(weight)
    
    def set_ensemble_method(self, method: str):
        """
        Set ensemble method.
        
        Args:
            method: 'average', 'weighted_average', 'voting', 'stacking'
        """
        self.ensemble_method = method
    
    def predict(self, input_data: Any) -> Any:
        """
        Make ensemble prediction.
        
        Args:
            input_data: Input data
            
        Returns:
            Ensemble prediction
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = [model.predict(input_data) for model in self.models]
        
        # Combine based on method
        if self.ensemble_method == 'average':
            import numpy as np
            return np.mean(predictions, axis=0)
        
        elif self.ensemble_method == 'weighted_average':
            import numpy as np
            total_weight = sum(self.weights)
            weighted_sum = sum(w * p for w, p in zip(self.weights, predictions))
            return weighted_sum / total_weight
        
        elif self.ensemble_method == 'voting':
            try:
                from scipy import stats
            except ImportError:
                raise ImportError(
                    "scipy is required for voting ensemble. Install it with: pip install scipy>=1.9.0"
                )
            import numpy as np
            # Majority voting
            predictions_array = np.array(predictions)
            return stats.mode(predictions_array, axis=0)[0]
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


# Convenience functions
def deploy_with_canary(
    deployment_manager: ModelDeploymentManager,
    model_name: str,
    model_version: str,
    initial_traffic: float = 5.0,
    target_traffic: float = 100.0,
    increment: float = 10.0
) -> bool:
    """
    Deploy model with canary strategy.
    
    Args:
        deployment_manager: Deployment manager
        model_name: Model name
        model_version: Model version
        initial_traffic: Initial traffic percentage
        target_traffic: Target traffic percentage
        increment: Increment per step
        
    Returns:
        True if successful
    """
    endpoint_id = deployment_manager.deploy_model(
        model_name=model_name,
        model_version=model_version,
        strategy=TrafficStrategy.CANARY,
        traffic_percentage=initial_traffic
    )
    
    return deployment_manager.canary_rollout(
        endpoint_id=endpoint_id,
        target_percentage=target_traffic,
        increment_percentage=increment
    )
