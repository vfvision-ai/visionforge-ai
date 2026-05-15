"""
Hyperparameter Optimizer - Automated hyperparameter tuning using advanced optimization.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from core.trainer import AutoTrainer
from utils.config import Config


class HyperparameterOptimizer:
    """Automated hyperparameter optimization using Optuna."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available. Install with: pip install optuna")
    
    def optimize(self, base_config: Config, n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization.
        
        Args:
            base_config: Base configuration to optimize
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, using default parameters")
            return self._get_default_params()
        
        self.logger.info(f"🔍 Starting hyperparameter optimization ({n_trials} trials)")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"optimization_{base_config.dataset_info.task_type}",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Store base config for objective function
        self.base_config = base_config
        
        # Run optimization
        study.optimize(self._objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"✅ Optimization complete. Best score: {best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        # Save optimization results
        self._save_optimization_results(study, base_config.output_dir)
        
        return best_params
    
    def _objective(self, trial) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy to maximize
        """
        # Sample hyperparameters based on task type
        params = self._sample_hyperparameters(trial)
        
        # Create config with sampled parameters
        config = self._create_config_with_params(self.base_config, params)
        
        # Reduce epochs for optimization (faster trials)
        config.max_epochs = min(20, config.max_epochs // 2)
        
        try:
            # Train model with sampled parameters
            trainer = AutoTrainer(config)
            results = trainer.train()
            
            # Return validation accuracy (or other metric to maximize)
            return results.best_accuracy
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {str(e)}")
            return 0.0  # Return poor score for failed trials
    
    def _sample_hyperparameters(self, trial) -> Dict[str, Any]:
        """Sample hyperparameters for the current trial."""
        task_type = self.base_config.dataset_info.task_type
        
        # Common parameters for all tasks
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'optimizer': trial.suggest_categorical('optimizer', ['adamw', 'sgd']),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'plateau'])
        }
        
        # Task-specific parameters
        if task_type == "classification":
            params.update({
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
                'mixup_alpha': trial.suggest_float('mixup_alpha', 0.0, 0.4),
            })
            
        elif task_type == "detection":
            params.update({
                'nms_threshold': trial.suggest_float('nms_threshold', 0.3, 0.7),
                'score_threshold': trial.suggest_float('score_threshold', 0.01, 0.1),
                'mosaic_prob': trial.suggest_float('mosaic_prob', 0.0, 0.8),
            })
            
        elif task_type == "segmentation":
            params.update({
                'deep_supervision': trial.suggest_categorical('deep_supervision', [True, False]),
                'auxiliary_loss_weight': trial.suggest_float('auxiliary_loss_weight', 0.1, 0.5),
            })
        
        return params
    
    def _create_config_with_params(self, base_config: Config, params: Dict[str, Any]) -> Config:
        """Create new config with optimized parameters."""
        # Create a copy of the base config
        config = Config.from_dict(base_config.to_dict())
        
        # Update with optimized parameters
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif key in ['dropout', 'label_smoothing', 'mixup_alpha', 'nms_threshold', 
                        'score_threshold', 'mosaic_prob', 'deep_supervision', 
                        'auxiliary_loss_weight']:
                # Update model config parameters
                config.model_config.config_params[key] = value
        
        return config
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Return default parameters when optimization is not available."""
        return {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 16,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'dropout': 0.1,
            'label_smoothing': 0.0
        }
    
    def _save_optimization_results(self, study, output_dir: Path):
        """Save optimization results to disk."""
        results_dir = output_dir / 'optimization'
        results_dir.mkdir(exist_ok=True)
        
        # Save study results
        results = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
        
        with open(results_dir / 'optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trials history
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })
        
        with open(results_dir / 'trials_history.json', 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        self.logger.info(f"Optimization results saved to: {results_dir}")


class GridSearchOptimizer:
    """Simple grid search optimizer as fallback."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, base_config: Config, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Perform grid search optimization.
        
        Args:
            base_config: Base configuration
            param_grid: Grid of parameters to search
            
        Returns:
            Best parameters found
        """
        self.logger.info("🔍 Starting grid search optimization")
        
        best_score = 0.0
        best_params = {}
        
        # Generate all parameter combinations
        param_combinations = self._generate_combinations(param_grid)
        
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            # Create config with current parameters
            config = self._create_config_with_params(base_config, params)
            config.max_epochs = 10  # Reduced for grid search
            
            try:
                # Train model
                trainer = AutoTrainer(config)
                results = trainer.train()
                
                # Check if this is the best score
                if results.best_accuracy > best_score:
                    best_score = results.best_accuracy
                    best_params = params.copy()
                    
            except Exception as e:
                self.logger.warning(f"Trial failed: {str(e)}")
                continue
        
        self.logger.info(f"✅ Grid search complete. Best score: {best_score:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def _generate_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
            
        return combinations
    
    def _create_config_with_params(self, base_config: Config, params: Dict[str, Any]) -> Config:
        """Create config with grid search parameters."""
        config = Config.from_dict(base_config.to_dict())
        
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        return config