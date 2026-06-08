"""
Advanced Analytics and Reporting System for VisionForge v2.2

Provides comprehensive analytics capabilities:
- Training analytics and visualizations
- Model performance comparison
- Automated report generation
- Statistical analysis
- Export to PDF/HTML/JSON
- Interactive dashboards
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import numpy as np
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsReport:
    """Represents an analytics report."""
    report_id: str
    title: str
    created_at: str
    report_type: str  # 'training', 'comparison', 'performance', 'summary'
    sections: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_id': self.report_id,
            'title': self.title,
            'created_at': self.created_at,
            'report_type': self.report_type,
            'sections': self.sections,
            'metadata': self.metadata
        }


class TrainingAnalytics:
    """
    Analyze training metrics and generate insights.
    """
    
    def __init__(self):
        """Initialize training analytics."""
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def record_metric(
        self,
        experiment_id: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Record metrics for an epoch.
        
        Args:
            experiment_id: Experiment identifier
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        self.metrics_history[experiment_id].append({
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
    
    def detect_overfitting(
        self,
        experiment_id: str,
        train_metric: str = 'train_loss',
        val_metric: str = 'val_loss',
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect overfitting by comparing train and validation metrics.
        
        Args:
            experiment_id: Experiment ID
            train_metric: Training metric name
            val_metric: Validation metric name
            threshold: Threshold for overfitting detection
            
        Returns:
            Overfitting analysis
        """
        if experiment_id not in self.metrics_history:
            return {'overfitting_detected': False, 'reason': 'No data'}
        
        history = self.metrics_history[experiment_id]
        
        # Extract metrics
        train_values = [h[train_metric] for h in history if train_metric in h]
        val_values = [h[val_metric] for h in history if val_metric in h]
        
        if not train_values or not val_values:
            return {'overfitting_detected': False, 'reason': 'Insufficient data'}
        
        # Calculate divergence
        recent_epochs = min(10, len(train_values))
        train_recent = np.mean(train_values[-recent_epochs:])
        val_recent = np.mean(val_values[-recent_epochs:])
        
        divergence = abs(val_recent - train_recent) / (train_recent + 1e-8)
        
        overfitting_detected = divergence > threshold and val_recent > train_recent
        
        return {
            'overfitting_detected': overfitting_detected,
            'divergence': float(divergence),
            'train_mean': float(train_recent),
            'val_mean': float(val_recent),
            'threshold': threshold,
            'recommendation': self._get_overfitting_recommendation(overfitting_detected, divergence)
        }
    
    def _get_overfitting_recommendation(self, overfitting: bool, divergence: float) -> str:
        """Get recommendation based on overfitting analysis."""
        if not overfitting:
            return "Training is healthy. Continue monitoring."
        
        if divergence > 0.3:
            return "Severe overfitting detected. Consider: stronger regularization, dropout, or early stopping."
        elif divergence > 0.2:
            return "Moderate overfitting. Try: data augmentation, reduce model complexity, or add dropout."
        else:
            return "Mild overfitting. Consider: light regularization or early stopping."
    
    def analyze_convergence(
        self,
        experiment_id: str,
        metric: str = 'val_loss',
        window_size: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze convergence behavior.
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to analyze
            window_size: Window for smoothing
            
        Returns:
            Convergence analysis
        """
        if experiment_id not in self.metrics_history:
            return {'converged': False, 'reason': 'No data'}
        
        history = self.metrics_history[experiment_id]
        values = [h[metric] for h in history if metric in h]
        
        if len(values) < window_size * 2:
            return {'converged': False, 'reason': 'Insufficient data'}
        
        # Calculate moving average
        smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        
        # Check if improvement has plateaued
        recent_window = smoothed[-window_size:]
        improvement = abs(recent_window[-1] - recent_window[0])
        relative_improvement = improvement / (abs(recent_window[0]) + 1e-8)
        
        converged = relative_improvement < 0.01  # Less than 1% improvement
        
        return {
            'converged': converged,
            'relative_improvement': float(relative_improvement),
            'current_value': float(values[-1]),
            'smoothed_value': float(smoothed[-1]),
            'trend': 'improving' if smoothed[-1] < smoothed[-min(5, len(smoothed))] else 'plateauing',
            'recommendation': 'Consider early stopping' if converged else 'Continue training'
        }
    
    def get_learning_rate_schedule_recommendation(
        self,
        experiment_id: str,
        metric: str = 'val_loss',
        patience: int = 5
    ) -> Dict[str, Any]:
        """
        Recommend learning rate schedule adjustments.
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to monitor
            patience: Patience epochs
            
        Returns:
            LR schedule recommendation
        """
        if experiment_id not in self.metrics_history:
            return {'action': 'none', 'reason': 'No data'}
        
        history = self.metrics_history[experiment_id]
        values = [h[metric] for h in history if metric in h]
        
        if len(values) < patience + 1:
            return {'action': 'none', 'reason': 'Insufficient epochs'}
        
        # Check if metric has improved in last N epochs
        recent = values[-patience:]
        best_recent = min(recent)
        previous_best = min(values[:-patience]) if len(values) > patience else float('inf')
        
        if best_recent >= previous_best:
            return {
                'action': 'reduce_lr',
                'reason': f'No improvement in last {patience} epochs',
                'current_best': float(best_recent),
                'previous_best': float(previous_best),
                'suggested_factor': 0.5
            }
        
        return {
            'action': 'none',
            'reason': 'Metric still improving',
            'current_best': float(best_recent)
        }


class ModelComparison:
    """
    Compare multiple models and experiments.
    """
    
    def __init__(self):
        """Initialize model comparison."""
        pass
    
    def compare_models(
        self,
        models: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models across metrics.
        
        Args:
            models: List of model metadata dictionaries
            metrics: Metrics to compare (if None, use all common metrics)
            
        Returns:
            Comparison results
        """
        if not models:
            return {'error': 'No models provided'}
        
        # Extract all available metrics
        all_metrics = set()
        for model in models:
            if 'metrics' in model:
                all_metrics.update(model['metrics'].keys())
        
        if metrics is None:
            metrics = list(all_metrics)
        
        # Build comparison table
        comparison = {
            'models': [],
            'metrics': metrics,
            'rankings': {},
            'best_model': None
        }
        
        for model in models:
            model_data = {
                'name': model.get('name', 'Unknown'),
                'version': model.get('version', 'Unknown'),
                'architecture': model.get('architecture', 'Unknown'),
                'metrics': {}
            }
            
            for metric in metrics:
                model_data['metrics'][metric] = model.get('metrics', {}).get(metric)
            
            comparison['models'].append(model_data)
        
        # Calculate rankings
        for metric in metrics:
            values = [(i, m['metrics'].get(metric)) for i, m in enumerate(comparison['models'])]
            # Filter out None values
            values = [(i, v) for i, v in values if v is not None]
            
            if not values:
                continue
            
            # Assume lower is better for loss metrics, higher for accuracy metrics
            reverse = 'acc' in metric.lower() or 'f1' in metric.lower() or 'precision' in metric.lower()
            ranked = sorted(values, key=lambda x: x[1], reverse=reverse)
            
            comparison['rankings'][metric] = [
                {
                    'rank': rank + 1,
                    'model_index': idx,
                    'model_name': comparison['models'][idx]['name'],
                    'value': val
                }
                for rank, (idx, val) in enumerate(ranked)
            ]
        
        # Determine best overall model (by average rank)
        if comparison['rankings']:
            model_ranks = defaultdict(list)
            for metric_ranks in comparison['rankings'].values():
                for entry in metric_ranks:
                    model_ranks[entry['model_index']].append(entry['rank'])
            
            avg_ranks = {idx: np.mean(ranks) for idx, ranks in model_ranks.items()}
            best_idx = min(avg_ranks.keys(), key=lambda k: avg_ranks[k])
            
            comparison['best_model'] = {
                'index': best_idx,
                'name': comparison['models'][best_idx]['name'],
                'average_rank': float(avg_ranks[best_idx])
            }
        
        return comparison
    
    def statistical_comparison(
        self,
        model1_results: List[float],
        model2_results: List[float],
        test_type: str = 't-test'
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison between two models.
        
        Args:
            model1_results: Results from model 1 (e.g., cross-validation scores)
            model2_results: Results from model 2
            test_type: Statistical test ('t-test', 'wilcoxon')
            
        Returns:
            Statistical test results
        """
        from scipy import stats
        
        if test_type == 't-test':
            statistic, p_value = stats.ttest_ind(model1_results, model2_results)
            test_name = "Independent t-test"
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(model1_results, model2_results)
            test_name = "Wilcoxon signed-rank test"
        else:
            return {'error': f'Unknown test type: {test_type}'}
        
        # Determine significance
        alpha = 0.05
        significant = p_value < alpha
        
        mean1, mean2 = np.mean(model1_results), np.mean(model2_results)
        
        return {
            'test': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'alpha': alpha,
            'model1_mean': float(mean1),
            'model2_mean': float(mean2),
            'difference': float(mean1 - mean2),
            'conclusion': self._get_statistical_conclusion(mean1, mean2, significant)
        }
    
    def _get_statistical_conclusion(self, mean1: float, mean2: float, significant: bool) -> str:
        """Generate conclusion from statistical test."""
        if not significant:
            return "No statistically significant difference between models."
        
        if mean1 > mean2:
            return "Model 1 performs significantly better than Model 2."
        else:
            return "Model 2 performs significantly better than Model 1."


class VisualizationGenerator:
    """
    Generate various visualizations for analytics.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualization generator.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self,
        history: List[Dict[str, Any]],
        metrics: List[str],
        title: str = "Training Curves",
        save_name: str = "training_curves.png"
    ) -> Path:
        """
        Plot training curves.
        
        Args:
            history: Training history
            metrics: Metrics to plot
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Path to saved plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        epochs = [h['epoch'] for h in history]
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            values = [h.get(metric, np.nan) for h in history]
            ax.plot(epochs, values, marker='o', label=metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} over Epochs')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves to {save_path}")
        return save_path
    
    def plot_model_comparison(
        self,
        comparison_data: Dict[str, Any],
        save_name: str = "model_comparison.png"
    ) -> Path:
        """
        Plot model comparison chart.
        
        Args:
            comparison_data: Comparison data from ModelComparison
            save_name: Filename to save
            
        Returns:
            Path to saved plot
        """
        if not MATPLOTLIB_AVAILABLE or not PANDAS_AVAILABLE:
            raise ImportError("Matplotlib and Pandas required")
        
        models = comparison_data['models']
        metrics = comparison_data['metrics']
        
        # Create DataFrame
        data = []
        for model in models:
            for metric in metrics:
                value = model['metrics'].get(metric)
                if value is not None:
                    data.append({
                        'Model': model['name'],
                        'Metric': metric,
                        'Value': value
                    })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if SEABORN_AVAILABLE:
            sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)
        else:
            # Fallback to matplotlib
            unique_metrics = df['Metric'].unique()
            unique_models = df['Model'].unique()
            x = np.arange(len(unique_metrics))
            width = 0.8 / len(unique_models)
            
            for i, model in enumerate(unique_models):
                model_data = df[df['Model'] == model]
                values = [model_data[model_data['Metric'] == m]['Value'].values[0] 
                         if len(model_data[model_data['Metric'] == m]) > 0 else 0
                         for m in unique_metrics]
                ax.bar(x + i * width, values, width, label=model)
            
            ax.set_xticks(x + width * (len(unique_models) - 1) / 2)
            ax.set_xticklabels(unique_metrics)
            ax.legend()
        
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison plot to {save_path}")
        return save_path
    
    def create_interactive_dashboard(
        self,
        history: List[Dict[str, Any]],
        metrics: List[str],
        save_name: str = "dashboard.html"
    ) -> Path:
        """
        Create interactive Plotly dashboard.
        
        Args:
            history: Training history
            metrics: Metrics to include
            save_name: Filename to save
            
        Returns:
            Path to saved dashboard
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for interactive dashboards")
        
        epochs = [h['epoch'] for h in history]
        
        # Create subplots
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics, 1):
            values = [h.get(metric, None) for h in history]
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=2),
                    marker=dict(size=6)
                ),
                row=i, col=1
            )
            
            fig.update_xaxis(title_text="Epoch", row=i, col=1)
            fig.update_yaxis(title_text=metric, row=i, col=1)
        
        fig.update_layout(
            title_text="Training Dashboard",
            showlegend=True,
            height=300 * len(metrics)
        )
        
        save_path = self.output_dir / save_name
        fig.write_html(str(save_path))
        
        logger.info(f"Saved interactive dashboard to {save_path}")
        return save_path


class ReportGenerator:
    """
    Generate comprehensive analytics reports.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_training_report(
        self,
        experiment_id: str,
        history: List[Dict[str, Any]],
        final_metrics: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> AnalyticsReport:
        """
        Generate comprehensive training report.
        
        Args:
            experiment_id: Experiment ID
            history: Training history
            final_metrics: Final metrics
            metadata: Additional metadata
            
        Returns:
            Analytics report
        """
        report = AnalyticsReport(
            report_id=f"training_{experiment_id}",
            title=f"Training Report: {metadata.get('name', experiment_id)}",
            created_at=datetime.now().isoformat(),
            report_type='training',
            metadata=metadata
        )
        
        # Summary section
        report.sections.append({
            'title': 'Summary',
            'content': {
                'experiment_id': experiment_id,
                'total_epochs': len(history),
                'architecture': metadata.get('architecture'),
                'dataset': metadata.get('dataset'),
                'final_metrics': final_metrics
            }
        })
        
        # Performance analysis
        analytics = TrainingAnalytics()
        for h in history:
            analytics.record_metric(experiment_id, h['epoch'], h)
        
        overfitting_analysis = analytics.detect_overfitting(experiment_id)
        convergence_analysis = analytics.analyze_convergence(experiment_id)
        
        report.sections.append({
            'title': 'Performance Analysis',
            'content': {
                'overfitting': overfitting_analysis,
                'convergence': convergence_analysis
            }
        })
        
        # Recommendations
        recommendations = []
        
        if overfitting_analysis.get('overfitting_detected'):
            recommendations.append(overfitting_analysis.get('recommendation'))
        
        if convergence_analysis.get('converged'):
            recommendations.append(convergence_analysis.get('recommendation'))
        
        report.sections.append({
            'title': 'Recommendations',
            'content': recommendations
        })
        
        return report
    
    def export_report_html(self, report: AnalyticsReport) -> Path:
        """
        Export report to HTML.
        
        Args:
            report: Analytics report
            
        Returns:
            Path to HTML file
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .section {{ margin-bottom: 30px; padding: 20px; background: #f9f9f9; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #fff; border: 1px solid #ddd; border-radius: 3px; }}
                .metric-name {{ font-weight: bold; color: #555; }}
                .metric-value {{ font-size: 1.2em; color: #2196F3; }}
                pre {{ background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>{report.title}</h1>
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {report.created_at}</p>
        """
        
        for section in report.sections:
            html_content += f"""
            <div class="section">
                <h2>{section['title']}</h2>
                <pre>{json.dumps(section['content'], indent=2)}</pre>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        save_path = self.output_dir / f"{report.report_id}.html"
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Exported HTML report to {save_path}")
        return save_path
    
    def export_report_json(self, report: AnalyticsReport) -> Path:
        """Export report to JSON."""
        save_path = self.output_dir / f"{report.report_id}.json"
        with open(save_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Exported JSON report to {save_path}")
        return save_path


# Convenience functions
def analyze_training(
    experiment_id: str,
    history: List[Dict[str, Any]],
    final_metrics: Dict[str, float],
    metadata: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    High-level function to analyze training and generate report.
    
    Args:
        experiment_id: Experiment ID
        history: Training history
        final_metrics: Final metrics
        metadata: Metadata
        output_dir: Output directory
        
    Returns:
        Analysis results with paths to generated files
    """
    # Generate visualizations
    viz_gen = VisualizationGenerator(output_dir)
    
    metrics_to_plot = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']
    available_metrics = [m for m in metrics_to_plot if any(m in h for h in history)]
    
    plot_path = viz_gen.plot_training_curves(history, available_metrics)
    dashboard_path = viz_gen.create_interactive_dashboard(history, available_metrics)
    
    # Generate report
    report_gen = ReportGenerator(output_dir)
    report = report_gen.generate_training_report(experiment_id, history, final_metrics, metadata)
    
    html_path = report_gen.export_report_html(report)
    json_path = report_gen.export_report_json(report)
    
    return {
        'report': report,
        'files': {
            'plot': str(plot_path),
            'dashboard': str(dashboard_path),
            'html_report': str(html_path),
            'json_report': str(json_path)
        }
    }
