"""
MLOps Automation for VisionForge v2.2

Provides end-to-end MLOps automation:
- Automated retraining pipelines
- Model performance degradation detection
- Automated model promotion
- CI/CD integration helpers
- Pipeline orchestration
- Feature store integration
- Model registry automation
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TriggerType(Enum):
    """Retraining trigger types."""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    DATA_AVAILABILITY = "data_availability"


@dataclass
class Pipeline:
    """ML pipeline configuration."""
    pipeline_id: str
    name: str
    steps: List[Dict[str, Any]]
    created_at: str
    status: PipelineStatus = PipelineStatus.PENDING
    last_run: Optional[str] = None
    run_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pipeline_id': self.pipeline_id,
            'name': self.name,
            'steps': self.steps,
            'created_at': self.created_at,
            'status': self.status.value,
            'last_run': self.last_run,
            'run_count': self.run_count,
            'metadata': self.metadata
        }


@dataclass
class RetrainingTrigger:
    """Automated retraining trigger configuration."""
    trigger_id: str
    trigger_type: TriggerType
    conditions: Dict[str, Any]
    pipeline_id: str
    enabled: bool = True
    last_triggered: Optional[str] = None
    trigger_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trigger_id': self.trigger_id,
            'trigger_type': self.trigger_type.value,
            'conditions': self.conditions,
            'pipeline_id': self.pipeline_id,
            'enabled': self.enabled,
            'last_triggered': self.last_triggered,
            'trigger_count': self.trigger_count
        }


class PipelineOrchestrator:
    """
    Orchestrate ML pipelines.
    """
    
    def __init__(self, pipelines_dir: Path):
        """
        Initialize pipeline orchestrator.
        
        Args:
            pipelines_dir: Directory for pipeline metadata
        """
        self.pipelines_dir = Path(pipelines_dir)
        self.pipelines_dir.mkdir(parents=True, exist_ok=True)
        
        self.pipelines_file = self.pipelines_dir / "pipelines.json"
        self.pipelines = self._load_pipelines()
        
        self.step_handlers: Dict[str, Callable] = {}
    
    def _load_pipelines(self) -> Dict[str, Pipeline]:
        """Load pipeline configurations."""
        if self.pipelines_file.exists():
            with open(self.pipelines_file, 'r') as f:
                data = json.load(f)
            pipelines = {}
            for k, v in data.items():
                v['status'] = PipelineStatus(v['status'])
                pipelines[k] = Pipeline(**v)
            return pipelines
        return {}
    
    def _save_pipelines(self):
        """Save pipeline configurations."""
        data = {k: v.to_dict() for k, v in self.pipelines.items()}
        with open(self.pipelines_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_step_handler(self, step_type: str, handler: Callable):
        """
        Register a handler for a pipeline step type.
        
        Args:
            step_type: Type of step (e.g., 'data_preparation', 'training', 'evaluation')
            handler: Handler function
        """
        self.step_handlers[step_type] = handler
        logger.info(f"Registered step handler: {step_type}")
    
    def create_pipeline(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new ML pipeline.
        
        Args:
            name: Pipeline name
            steps: List of pipeline steps
            metadata: Additional metadata
            
        Returns:
            Pipeline ID
        """
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=name,
            steps=steps,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.pipelines[pipeline_id] = pipeline
        self._save_pipelines()
        
        logger.info(f"Created pipeline: {pipeline_id}")
        return pipeline_id
    
    def run_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Execute a pipeline.
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            Execution results
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = PipelineStatus.RUNNING
        pipeline.last_run = datetime.now().isoformat()
        pipeline.run_count += 1
        self._save_pipelines()
        
        results = {
            'pipeline_id': pipeline_id,
            'start_time': pipeline.last_run,
            'steps': []
        }
        
        try:
            # Execute each step
            context = {}  # Shared context between steps
            
            for i, step in enumerate(pipeline.steps):
                step_type = step.get('type')
                step_config = step.get('config', {})
                
                logger.info(f"Executing step {i+1}/{len(pipeline.steps)}: {step_type}")
                
                if step_type not in self.step_handlers:
                    raise ValueError(f"No handler registered for step type: {step_type}")
                
                # Execute step
                handler = self.step_handlers[step_type]
                step_result = handler(step_config, context)
                
                results['steps'].append({
                    'step': i + 1,
                    'type': step_type,
                    'status': 'success',
                    'result': step_result
                })
                
                # Update context
                if isinstance(step_result, dict):
                    context.update(step_result)
            
            pipeline.status = PipelineStatus.SUCCESS
            results['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline.status = PipelineStatus.FAILED
            results['status'] = 'failed'
            results['error'] = str(e)
        
        finally:
            results['end_time'] = datetime.now().isoformat()
            self._save_pipelines()
        
        logger.info(f"Pipeline {pipeline_id} completed with status: {pipeline.status.value}")
        return results
    
    def get_pipeline_status(self, pipeline_id: str) -> PipelineStatus:
        """Get current pipeline status."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        return self.pipelines[pipeline_id].status
    
    def list_pipelines(self) -> List[Pipeline]:
        """List all pipelines."""
        return list(self.pipelines.values())


class AutomatedRetrainingManager:
    """
    Manage automated model retraining based on triggers.
    """
    
    def __init__(
        self,
        triggers_dir: Path,
        orchestrator: PipelineOrchestrator
    ):
        """
        Initialize retraining manager.
        
        Args:
            triggers_dir: Directory for trigger configurations
            orchestrator: Pipeline orchestrator
        """
        self.triggers_dir = Path(triggers_dir)
        self.triggers_dir.mkdir(parents=True, exist_ok=True)
        
        self.triggers_file = self.triggers_dir / "triggers.json"
        self.triggers = self._load_triggers()
        
        self.orchestrator = orchestrator
    
    def _load_triggers(self) -> Dict[str, RetrainingTrigger]:
        """Load trigger configurations."""
        if self.triggers_file.exists():
            with open(self.triggers_file, 'r') as f:
                data = json.load(f)
            triggers = {}
            for k, v in data.items():
                v['trigger_type'] = TriggerType(v['trigger_type'])
                triggers[k] = RetrainingTrigger(**v)
            return triggers
        return {}
    
    def _save_triggers(self):
        """Save trigger configurations."""
        data = {k: v.to_dict() for k, v in self.triggers.items()}
        with open(self.triggers_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_scheduled_trigger(
        self,
        pipeline_id: str,
        interval_hours: int = 24,
        start_time: Optional[str] = None
    ) -> str:
        """
        Create scheduled retraining trigger.
        
        Args:
            pipeline_id: Pipeline to trigger
            interval_hours: Trigger interval in hours
            start_time: Start time (ISO format)
            
        Returns:
            Trigger ID
        """
        trigger_id = f"trigger_scheduled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trigger = RetrainingTrigger(
            trigger_id=trigger_id,
            trigger_type=TriggerType.SCHEDULED,
            conditions={
                'interval_hours': interval_hours,
                'start_time': start_time or datetime.now().isoformat()
            },
            pipeline_id=pipeline_id
        )
        
        self.triggers[trigger_id] = trigger
        self._save_triggers()
        
        logger.info(f"Created scheduled trigger: {trigger_id}")
        return trigger_id
    
    def create_performance_trigger(
        self,
        pipeline_id: str,
        metric_name: str,
        threshold: float,
        comparison: str = 'less_than'
    ) -> str:
        """
        Create performance degradation trigger.
        
        Args:
            pipeline_id: Pipeline to trigger
            metric_name: Metric to monitor
            threshold: Threshold value
            comparison: 'less_than', 'greater_than', 'equals'
            
        Returns:
            Trigger ID
        """
        trigger_id = f"trigger_perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trigger = RetrainingTrigger(
            trigger_id=trigger_id,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            conditions={
                'metric_name': metric_name,
                'threshold': threshold,
                'comparison': comparison
            },
            pipeline_id=pipeline_id
        )
        
        self.triggers[trigger_id] = trigger
        self._save_triggers()
        
        logger.info(f"Created performance trigger: {trigger_id}")
        return trigger_id
    
    def create_drift_trigger(
        self,
        pipeline_id: str,
        drift_threshold: float = 0.25,
        check_interval_hours: int = 6
    ) -> str:
        """
        Create data drift trigger.
        
        Args:
            pipeline_id: Pipeline to trigger
            drift_threshold: PSI threshold for drift detection
            check_interval_hours: How often to check for drift
            
        Returns:
            Trigger ID
        """
        trigger_id = f"trigger_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trigger = RetrainingTrigger(
            trigger_id=trigger_id,
            trigger_type=TriggerType.DATA_DRIFT,
            conditions={
                'drift_threshold': drift_threshold,
                'check_interval_hours': check_interval_hours
            },
            pipeline_id=pipeline_id
        )
        
        self.triggers[trigger_id] = trigger
        self._save_triggers()
        
        logger.info(f"Created drift trigger: {trigger_id}")
        return trigger_id
    
    def check_triggers(self, context: Dict[str, Any]) -> List[str]:
        """
        Check all triggers and execute triggered pipelines.
        
        Args:
            context: Context with current metrics, drift scores, etc.
            
        Returns:
            List of triggered pipeline IDs
        """
        triggered = []
        
        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue
            
            should_trigger = False
            
            if trigger.trigger_type == TriggerType.SCHEDULED:
                should_trigger = self._check_scheduled_trigger(trigger)
            
            elif trigger.trigger_type == TriggerType.PERFORMANCE_DEGRADATION:
                should_trigger = self._check_performance_trigger(trigger, context)
            
            elif trigger.trigger_type == TriggerType.DATA_DRIFT:
                should_trigger = self._check_drift_trigger(trigger, context)
            
            if should_trigger:
                logger.info(f"Trigger {trigger.trigger_id} activated, executing pipeline {trigger.pipeline_id}")
                
                try:
                    self.orchestrator.run_pipeline(trigger.pipeline_id)
                    trigger.last_triggered = datetime.now().isoformat()
                    trigger.trigger_count += 1
                    triggered.append(trigger.pipeline_id)
                except Exception as e:
                    logger.error(f"Failed to execute pipeline {trigger.pipeline_id}: {e}")
        
        if triggered:
            self._save_triggers()
        
        return triggered
    
    def _check_scheduled_trigger(self, trigger: RetrainingTrigger) -> bool:
        """Check if scheduled trigger should fire."""
        if not trigger.last_triggered:
            return True
        
        last_run = datetime.fromisoformat(trigger.last_triggered)
        interval = timedelta(hours=trigger.conditions['interval_hours'])
        
        return datetime.now() >= last_run + interval
    
    def _check_performance_trigger(
        self,
        trigger: RetrainingTrigger,
        context: Dict[str, Any]
    ) -> bool:
        """Check if performance trigger should fire."""
        metric_name = trigger.conditions['metric_name']
        threshold = trigger.conditions['threshold']
        comparison = trigger.conditions['comparison']
        
        if metric_name not in context:
            return False
        
        current_value = context[metric_name]
        
        if comparison == 'less_than':
            return current_value < threshold
        elif comparison == 'greater_than':
            return current_value > threshold
        elif comparison == 'equals':
            return abs(current_value - threshold) < 1e-6
        
        return False
    
    def _check_drift_trigger(
        self,
        trigger: RetrainingTrigger,
        context: Dict[str, Any]
    ) -> bool:
        """Check if drift trigger should fire."""
        drift_threshold = trigger.conditions['drift_threshold']
        
        drift_score = context.get('drift_score', 0)
        return drift_score > drift_threshold


class ModelPromotionAutomation:
    """
    Automate model promotion through stages.
    """
    
    def __init__(self):
        """Initialize model promotion automation."""
        self.promotion_rules: List[Dict[str, Any]] = []
    
    def add_promotion_rule(
        self,
        from_stage: str,
        to_stage: str,
        conditions: Dict[str, Any]
    ):
        """
        Add model promotion rule.
        
        Args:
            from_stage: Source stage
            to_stage: Target stage
            conditions: Promotion conditions
        """
        rule = {
            'from_stage': from_stage,
            'to_stage': to_stage,
            'conditions': conditions
        }
        self.promotion_rules.append(rule)
        logger.info(f"Added promotion rule: {from_stage} -> {to_stage}")
    
    def evaluate_promotion(
        self,
        model_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Evaluate if model should be promoted.
        
        Args:
            model_metadata: Model metadata including metrics
            
        Returns:
            Target stage if promotion criteria met, None otherwise
        """
        current_stage = model_metadata.get('stage')
        metrics = model_metadata.get('metrics', {})
        
        for rule in self.promotion_rules:
            if rule['from_stage'] != current_stage:
                continue
            
            # Check all conditions
            conditions_met = True
            for metric_name, required_value in rule['conditions'].items():
                if metric_name not in metrics:
                    conditions_met = False
                    break
                
                # Simple threshold check (can be extended)
                if metrics[metric_name] < required_value:
                    conditions_met = False
                    break
            
            if conditions_met:
                logger.info(f"Model promotion criteria met: {current_stage} -> {rule['to_stage']}")
                return rule['to_stage']
        
        return None


class CICDIntegration:
    """
    Helpers for CI/CD pipeline integration.
    """
    
    @staticmethod
    def generate_training_job_config(
        model_name: str,
        dataset_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Generate configuration for CI/CD training job.
        
        Args:
            model_name: Model name
            dataset_path: Dataset path
            output_dir: Output directory
            
        Returns:
            Job configuration
        """
        return {
            'job_name': f"train_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model_name': model_name,
            'dataset_path': dataset_path,
            'output_dir': output_dir,
            'timestamp': datetime.now().isoformat(),
            'steps': [
                {'name': 'data_validation', 'type': 'validation'},
                {'name': 'model_training', 'type': 'training'},
                {'name': 'model_evaluation', 'type': 'evaluation'},
                {'name': 'model_registration', 'type': 'registration'}
            ]
        }
    
    @staticmethod
    def generate_deployment_config(
        model_name: str,
        model_version: str,
        environment: str = 'production'
    ) -> Dict[str, Any]:
        """
        Generate deployment configuration.
        
        Args:
            model_name: Model name
            model_version: Model version
            environment: Target environment
            
        Returns:
            Deployment configuration
        """
        return {
            'deployment_name': f"deploy_{model_name}_v{model_version}",
            'model_name': model_name,
            'model_version': model_version,
            'environment': environment,
            'strategy': 'canary',
            'initial_traffic': 5,
            'health_check': {
                'enabled': True,
                'endpoint': '/health',
                'interval': 30
            },
            'rollback': {
                'enabled': True,
                'trigger_on_error_rate': 0.05
            }
        }


# Convenience functions
def create_retraining_pipeline(
    orchestrator: PipelineOrchestrator,
    model_name: str,
    dataset_path: str
) -> str:
    """
    Create standard retraining pipeline.
    
    Args:
        orchestrator: Pipeline orchestrator
        model_name: Model to retrain
        dataset_path: Dataset path
        
    Returns:
        Pipeline ID
    """
    steps = [
        {
            'type': 'data_preparation',
            'config': {'dataset_path': dataset_path, 'model_name': model_name}
        },
        {
            'type': 'training',
            'config': {'model_name': model_name}
        },
        {
            'type': 'evaluation',
            'config': {'model_name': model_name}
        },
        {
            'type': 'registration',
            'config': {'model_name': model_name}
        }
    ]
    
    return orchestrator.create_pipeline(
        name=f"Retraining Pipeline: {model_name}",
        steps=steps,
        metadata={'model_name': model_name, 'dataset_path': dataset_path}
    )
