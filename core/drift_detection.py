"""
Data Drift Detection and Monitoring for VisionForge v2.2

Detects and monitors data drift in production:
- Statistical drift detection (KS test, Chi-square)
- Feature distribution monitoring
- Covariate shift detection
- Concept drift detection
- Automated alerts on drift
- Visualization of drift metrics
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of data drift."""
    COVARIATE_SHIFT = "covariate_shift"  # Input distribution changes
    PRIOR_PROBABILITY_SHIFT = "prior_probability_shift"  # Label distribution changes
    CONCEPT_DRIFT = "concept_drift"  # Relationship between X and y changes


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftReport:
    """Drift detection report."""
    timestamp: str
    drift_detected: bool
    drift_type: Optional[DriftType]
    severity: DriftSeverity
    metrics: Dict[str, float]
    affected_features: List[str] = field(default_factory=list)
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'drift_detected': self.drift_detected,
            'drift_type': self.drift_type.value if self.drift_type else None,
            'severity': self.severity.value,
            'metrics': self.metrics,
            'affected_features': self.affected_features,
            'recommendation': self.recommendation
        }


class DataDriftDetector:
    """
    Detect various types of data drift.
    """
    
    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        significance_level: float = 0.05
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference/baseline data
            significance_level: Statistical significance level
        """
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.feature_statistics: Dict[str, Dict[str, float]] = {}
        
        if reference_data is not None:
            self._compute_reference_statistics()
    
    def _compute_reference_statistics(self):
        """Compute statistics for reference data."""
        if self.reference_data.ndim == 1:
            data = self.reference_data.reshape(-1, 1)
        else:
            data = self.reference_data
        
        for i in range(data.shape[1]):
            feature_data = data[:, i]
            self.feature_statistics[f'feature_{i}'] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'median': float(np.median(feature_data)),
                'q25': float(np.percentile(feature_data, 25)),
                'q75': float(np.percentile(feature_data, 75))
            }
    
    def detect_drift_ks_test(
        self,
        current_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            current_data: Current data to compare
            feature_names: Optional feature names
            
        Returns:
            Drift report
        """
        try:
            from scipy import stats
        except ImportError:
            raise ImportError(
                "scipy is required for drift detection. Install it with: pip install scipy>=1.9.0"
            )
        
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        if self.reference_data.ndim == 1:
            ref_data = self.reference_data.reshape(-1, 1)
        else:
            ref_data = self.reference_data
        
        if current_data.ndim == 1:
            curr_data = current_data.reshape(-1, 1)
        else:
            curr_data = current_data
        
        num_features = min(ref_data.shape[1], curr_data.shape[1])
        
        drift_detected = False
        affected_features = []
        p_values = {}
        statistics = {}
        
        for i in range(num_features):
            feature_name = feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'
            
            # Perform KS test
            ks_stat, p_value = stats.ks_2samp(ref_data[:, i], curr_data[:, i])
            
            p_values[feature_name] = float(p_value)
            statistics[feature_name] = float(ks_stat)
            
            if p_value < self.significance_level:
                drift_detected = True
                affected_features.append(feature_name)
        
        # Determine severity
        severity = self._determine_severity(p_values)
        
        # Generate recommendation
        recommendation = self._generate_drift_recommendation(
            drift_detected, severity, affected_features
        )
        
        return DriftReport(
            timestamp=datetime.now().isoformat(),
            drift_detected=drift_detected,
            drift_type=DriftType.COVARIATE_SHIFT if drift_detected else None,
            severity=severity,
            metrics={
                'p_values': p_values,
                'ks_statistics': statistics,
                'significance_level': self.significance_level
            },
            affected_features=affected_features,
            recommendation=recommendation
        )
    
    def detect_drift_psi(
        self,
        current_data: np.ndarray,
        num_bins: int = 10,
        feature_names: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Detect drift using Population Stability Index (PSI).
        
        Args:
            current_data: Current data
            num_bins: Number of bins for discretization
            feature_names: Feature names
            
        Returns:
            Drift report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        if self.reference_data.ndim == 1:
            ref_data = self.reference_data.reshape(-1, 1)
        else:
            ref_data = self.reference_data
        
        if current_data.ndim == 1:
            curr_data = current_data.reshape(-1, 1)
        else:
            curr_data = current_data
        
        num_features = min(ref_data.shape[1], curr_data.shape[1])
        
        psi_values = {}
        affected_features = []
        
        for i in range(num_features):
            feature_name = feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'
            
            # Calculate PSI
            psi = self._calculate_psi(ref_data[:, i], curr_data[:, i], num_bins)
            psi_values[feature_name] = float(psi)
            
            # PSI thresholds: < 0.1 (no drift), 0.1-0.25 (moderate), > 0.25 (significant)
            if psi > 0.1:
                affected_features.append(feature_name)
        
        # Determine overall drift
        max_psi = max(psi_values.values()) if psi_values else 0
        drift_detected = max_psi > 0.1
        
        severity = self._determine_severity_from_psi(max_psi)
        recommendation = self._generate_drift_recommendation(
            drift_detected, severity, affected_features
        )
        
        return DriftReport(
            timestamp=datetime.now().isoformat(),
            drift_detected=drift_detected,
            drift_type=DriftType.COVARIATE_SHIFT if drift_detected else None,
            severity=severity,
            metrics={'psi_values': psi_values, 'max_psi': float(max_psi)},
            affected_features=affected_features,
            recommendation=recommendation
        )
    
    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        num_bins: int
    ) -> float:
        """
        Calculate Population Stability Index.
        
        Args:
            reference: Reference data
            current: Current data
            num_bins: Number of bins
            
        Returns:
            PSI value
        """
        # Create bins based on reference data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, num_bins + 1)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bins)
        curr_counts, _ = np.histogram(current, bins=bins)
        
        # Convert to percentages
        ref_percents = ref_counts / len(reference)
        curr_percents = curr_counts / len(current)
        
        # Avoid division by zero
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)
        
        # Calculate PSI
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        
        return psi
    
    def _determine_severity(self, p_values: Dict[str, float]) -> DriftSeverity:
        """Determine drift severity from p-values."""
        if not p_values:
            return DriftSeverity.NONE
        
        min_p = min(p_values.values())
        num_drifted = sum(1 for p in p_values.values() if p < self.significance_level)
        drift_ratio = num_drifted / len(p_values)
        
        if min_p >= self.significance_level:
            return DriftSeverity.NONE
        elif drift_ratio < 0.2 and min_p > 0.01:
            return DriftSeverity.LOW
        elif drift_ratio < 0.5 and min_p > 0.001:
            return DriftSeverity.MEDIUM
        elif drift_ratio < 0.7:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _determine_severity_from_psi(self, max_psi: float) -> DriftSeverity:
        """Determine severity from PSI value."""
        if max_psi < 0.1:
            return DriftSeverity.NONE
        elif max_psi < 0.15:
            return DriftSeverity.LOW
        elif max_psi < 0.25:
            return DriftSeverity.MEDIUM
        elif max_psi < 0.5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _generate_drift_recommendation(
        self,
        drift_detected: bool,
        severity: DriftSeverity,
        affected_features: List[str]
    ) -> str:
        """Generate recommendation based on drift detection."""
        if not drift_detected:
            return "No drift detected. Continue monitoring."
        
        if severity == DriftSeverity.LOW:
            return f"Low drift detected in {len(affected_features)} feature(s). Monitor closely."
        elif severity == DriftSeverity.MEDIUM:
            return f"Moderate drift in {len(affected_features)} feature(s). Consider retraining model."
        elif severity == DriftSeverity.HIGH:
            return f"High drift in {len(affected_features)} feature(s). Retrain model soon."
        else:  # CRITICAL
            return f"Critical drift in {len(affected_features)} feature(s). Immediate retraining required!"
    
    def detect_concept_drift(
        self,
        current_predictions: np.ndarray,
        current_labels: np.ndarray,
        window_size: int = 100
    ) -> Dict[str, Any]:
        """
        Detect concept drift by monitoring prediction accuracy.
        
        Args:
            current_predictions: Model predictions
            current_labels: True labels
            window_size: Window size for moving average
            
        Returns:
            Concept drift report
        """
        # Calculate accuracy over sliding window
        accuracies = []
        for i in range(len(current_predictions) - window_size + 1):
            window_preds = current_predictions[i:i+window_size]
            window_labels = current_labels[i:i+window_size]
            accuracy = np.mean(window_preds == window_labels)
            accuracies.append(accuracy)
        
        if not accuracies:
            return {'concept_drift_detected': False, 'reason': 'Insufficient data'}
        
        # Detect significant drops in accuracy
        accuracies = np.array(accuracies)
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        # Check recent performance
        recent_window = min(10, len(accuracies) // 2)
        recent_accuracy = np.mean(accuracies[-recent_window:])
        
        # Detect drift if recent accuracy drops significantly
        drift_threshold = mean_accuracy - 2 * std_accuracy
        concept_drift = recent_accuracy < drift_threshold
        
        return {
            'concept_drift_detected': concept_drift,
            'mean_accuracy': float(mean_accuracy),
            'recent_accuracy': float(recent_accuracy),
            'threshold': float(drift_threshold),
            'performance_drop': float(mean_accuracy - recent_accuracy),
            'recommendation': 'Retrain model with recent data' if concept_drift else 'Performance stable'
        }


class DriftMonitor:
    """
    Continuous monitoring of data drift.
    """
    
    def __init__(self, monitor_dir: Path):
        """
        Initialize drift monitor.
        
        Args:
            monitor_dir: Directory for monitoring data
        """
        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_file = self.monitor_dir / "drift_reports.json"
        self.reports: List[DriftReport] = []
        self._load_reports()
        
        self.detectors: Dict[str, DataDriftDetector] = {}
    
    def _load_reports(self):
        """Load historical reports."""
        if self.reports_file.exists():
            with open(self.reports_file, 'r') as f:
                data = json.load(f)
            # Convert back to DriftReport objects (simplified)
            self.reports = data  # Store as list of dicts for now
    
    def _save_reports(self):
        """Save reports to disk."""
        reports_data = [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.reports]
        with open(self.reports_file, 'w') as f:
            json.dump(reports_data, f, indent=2)
    
    def register_detector(
        self,
        name: str,
        reference_data: np.ndarray,
        significance_level: float = 0.05
    ):
        """
        Register a drift detector.
        
        Args:
            name: Detector name
            reference_data: Reference data
            significance_level: Significance level
        """
        self.detectors[name] = DataDriftDetector(reference_data, significance_level)
        logger.info(f"Registered drift detector: {name}")
    
    def check_drift(
        self,
        detector_name: str,
        current_data: np.ndarray,
        method: str = 'ks_test',
        feature_names: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Check for drift using specified method.
        
        Args:
            detector_name: Name of detector to use
            current_data: Current data
            method: Detection method ('ks_test', 'psi')
            feature_names: Feature names
            
        Returns:
            Drift report
        """
        if detector_name not in self.detectors:
            raise ValueError(f"Detector {detector_name} not registered")
        
        detector = self.detectors[detector_name]
        
        if method == 'ks_test':
            report = detector.detect_drift_ks_test(current_data, feature_names)
        elif method == 'psi':
            report = detector.detect_drift_psi(current_data, feature_names=feature_names)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store report
        self.reports.append(report)
        self._save_reports()
        
        # Log alert if drift detected
        if report.drift_detected:
            logger.warning(
                f"Drift detected! Severity: {report.severity.value}, "
                f"Affected features: {len(report.affected_features)}"
            )
        
        return report
    
    def get_drift_history(
        self,
        hours: Optional[int] = None,
        severity: Optional[DriftSeverity] = None
    ) -> List[DriftReport]:
        """
        Get drift history with optional filtering.
        
        Args:
            hours: Only reports from last N hours
            severity: Filter by severity
            
        Returns:
            List of drift reports
        """
        reports = self.reports
        
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            reports = [
                r for r in reports
                if datetime.fromisoformat(r.timestamp if hasattr(r, 'timestamp') else r['timestamp']) >= cutoff
            ]
        
        if severity:
            severity_val = severity.value
            reports = [
                r for r in reports
                if (r.severity.value if hasattr(r, 'severity') else r['severity']) == severity_val
            ]
        
        return reports
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get summary of drift monitoring.
        
        Returns:
            Summary statistics
        """
        if not self.reports:
            return {'total_checks': 0, 'drift_detected': 0}
        
        total = len(self.reports)
        drifted = sum(
            1 for r in self.reports
            if (r.drift_detected if hasattr(r, 'drift_detected') else r['drift_detected'])
        )
        
        # Count by severity
        severity_counts = {}
        for r in self.reports:
            sev = r.severity.value if hasattr(r, 'severity') else r['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            'total_checks': total,
            'drift_detected': drifted,
            'drift_rate': drifted / total if total > 0 else 0,
            'severity_distribution': severity_counts,
            'latest_check': self.reports[-1].timestamp if hasattr(self.reports[-1], 'timestamp') else self.reports[-1]['timestamp']
        }


# Convenience functions
def setup_drift_monitoring(
    reference_data: np.ndarray,
    monitor_dir: Path,
    detector_name: str = "default"
) -> DriftMonitor:
    """
    Set up drift monitoring with reference data.
    
    Args:
        reference_data: Reference/baseline data
        monitor_dir: Directory for monitoring
        detector_name: Name for detector
        
    Returns:
        Configured drift monitor
    """
    monitor = DriftMonitor(monitor_dir)
    monitor.register_detector(detector_name, reference_data)
    return monitor
