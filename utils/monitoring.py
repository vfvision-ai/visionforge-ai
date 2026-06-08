"""
Enhanced Monitoring, Alerting, and Health Checks for VisionForge v2.1

Provides comprehensive system monitoring:
- Real-time metrics collection (Prometheus compatible)
- Custom alerting with thresholds
- Health check endpoints with dependency checks
- Performance monitoring and profiling
- Resource usage tracking (CPU, GPU, memory, disk)
- Training job monitoring
- Model serving metrics
"""

import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import json
import threading

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Represents a single metric."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'unit': self.unit
        }


@dataclass
class Alert:
    """Represents an alert."""
    id: str
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'id': self.id,
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp,
            'resolved': self.resolved
        }
        if self.metric_name:
            data['metric_name'] = self.metric_name
        if self.metric_value is not None:
            data['metric_value'] = self.metric_value
        if self.threshold is not None:
            data['threshold'] = self.threshold
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at
        return data


class MetricsCollector:
    """
    Collect and store system metrics.
    Compatible with Prometheus exposition format.
    """
    
    def __init__(self, retention_seconds: int = 3600):
        """
        Initialize metrics collector.
        
        Args:
            retention_seconds: How long to keep metrics
        """
        self.retention_seconds = retention_seconds
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._lock = threading.Lock()
    
    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        unit: str = ""
    ):
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            unit: Unit of measurement
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            unit=unit
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def get_latest(self, name: str) -> Optional[Metric]:
        """Get latest value for a metric."""
        with self._lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1]
        return None
    
    def get_history(
        self,
        name: str,
        seconds: Optional[int] = None
    ) -> List[Metric]:
        """
        Get metric history.
        
        Args:
            name: Metric name
            seconds: Only return metrics from last N seconds
            
        Returns:
            List of metrics
        """
        with self._lock:
            if name not in self.metrics:
                return []
            
            metrics = list(self.metrics[name])
        
        if seconds is not None:
            cutoff = time.time() - seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        return metrics
    
    def get_average(self, name: str, seconds: int = 60) -> Optional[float]:
        """Get average value over time window."""
        history = self.get_history(name, seconds)
        if not history:
            return None
        return sum(m.value for m in history) / len(history)
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff = time.time() - self.retention_seconds
        
        with self._lock:
            for name in self.metrics:
                # Remove old metrics
                while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
                    self.metrics[name].popleft()
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics
        """
        lines = []
        
        with self._lock:
            for name, metric_list in self.metrics.items():
                if not metric_list:
                    continue
                
                latest = metric_list[-1]
                
                # Add help and type
                lines.append(f"# HELP {name} {name}")
                lines.append(f"# TYPE {name} gauge")
                
                # Add metric with labels
                if latest.labels:
                    label_str = ','.join(f'{k}="{v}"' for k, v in latest.labels.items())
                    lines.append(f"{name}{{{label_str}}} {latest.value}")
                else:
                    lines.append(f"{name} {latest.value}")
        
        return '\n'.join(lines)


class SystemMonitor:
    """
    Monitor system resources (CPU, memory, GPU, disk).
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize system monitor.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def collect_system_metrics(self):
        """Collect current system metrics."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics.record('system_cpu_percent', cpu_percent, unit='percent')
        
        cpu_count = psutil.cpu_count()
        self.metrics.record('system_cpu_count', cpu_count)
        
        # Memory
        memory = psutil.virtual_memory()
        self.metrics.record('system_memory_used_bytes', memory.used, unit='bytes')
        self.metrics.record('system_memory_total_bytes', memory.total, unit='bytes')
        self.metrics.record('system_memory_percent', memory.percent, unit='percent')
        
        # Disk
        disk = psutil.disk_usage('/')
        self.metrics.record('system_disk_used_bytes', disk.used, unit='bytes')
        self.metrics.record('system_disk_total_bytes', disk.total, unit='bytes')
        self.metrics.record('system_disk_percent', disk.percent, unit='percent')
        
        # GPU (if available)
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    labels = {'gpu_id': str(i), 'gpu_name': gpu.name}
                    self.metrics.record('gpu_utilization_percent', gpu.load * 100, labels, 'percent')
                    self.metrics.record('gpu_memory_used_mb', gpu.memoryUsed, labels, 'MB')
                    self.metrics.record('gpu_memory_total_mb', gpu.memoryTotal, labels, 'MB')
                    self.metrics.record('gpu_temperature_celsius', gpu.temperature, labels, 'C')
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
    
    def start_monitoring(self, interval_seconds: int = 10):
        """
        Start continuous monitoring in background thread.
        
        Args:
            interval_seconds: Monitoring interval
        """
        if self._monitoring:
            logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                try:
                    self.collect_system_metrics()
                except Exception as e:
                    logger.error(f"Error collecting metrics: {e}")
                
                time.sleep(interval_seconds)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started system monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped system monitoring")


class AlertManager:
    """
    Manage alerts based on metric thresholds.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_handlers: Optional[List[Callable]] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
            alert_handlers: Functions to call when alerts fire
        """
        self.metrics = metrics_collector
        self.alert_handlers = alert_handlers or []
        self.alerts: Dict[str, Alert] = {}
        self.rules: List[Dict[str, Any]] = []
        self._checking = False
        self._check_thread: Optional[threading.Thread] = None
    
    def add_rule(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        operator: str = '>',
        level: AlertLevel = AlertLevel.WARNING,
        message_template: str = "{metric} is {value} (threshold: {threshold})"
    ):
        """
        Add alerting rule.
        
        Args:
            name: Rule name
            metric_name: Metric to check
            threshold: Threshold value
            operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
            level: Alert level
            message_template: Alert message template
        """
        rule = {
            'name': name,
            'metric_name': metric_name,
            'threshold': threshold,
            'operator': operator,
            'level': level,
            'message_template': message_template
        }
        self.rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    def check_rules(self):
        """Check all rules and fire alerts if needed."""
        for rule in self.rules:
            metric = self.metrics.get_latest(rule['metric_name'])
            if metric is None:
                continue
            
            # Check condition
            triggered = self._evaluate_condition(
                metric.value,
                rule['threshold'],
                rule['operator']
            )
            
            alert_id = f"{rule['name']}_{rule['metric_name']}"
            
            if triggered:
                # Fire alert if not already active
                if alert_id not in self.alerts or self.alerts[alert_id].resolved:
                    alert = Alert(
                        id=alert_id,
                        level=rule['level'],
                        message=rule['message_template'].format(
                            metric=rule['metric_name'],
                            value=metric.value,
                            threshold=rule['threshold']
                        ),
                        timestamp=time.time(),
                        metric_name=rule['metric_name'],
                        metric_value=metric.value,
                        threshold=rule['threshold']
                    )
                    self.alerts[alert_id] = alert
                    self._fire_alert(alert)
            else:
                # Resolve alert if active
                if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                    self.alerts[alert_id].resolved = True
                    self.alerts[alert_id].resolved_at = time.time()
                    logger.info(f"Resolved alert: {alert_id}")
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate threshold condition."""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def _fire_alert(self, alert: Alert):
        """Fire an alert and call handlers."""
        logger.log(
            logging.CRITICAL if alert.level == AlertLevel.CRITICAL else logging.WARNING,
            f"ALERT [{alert.level.value.upper()}]: {alert.message}"
        )
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def start_checking(self, interval_seconds: int = 30):
        """
        Start continuous rule checking.
        
        Args:
            interval_seconds: Check interval
        """
        if self._checking:
            logger.warning("Alert checking already started")
            return
        
        self._checking = True
        
        def check_loop():
            while self._checking:
                try:
                    self.check_rules()
                except Exception as e:
                    logger.error(f"Error checking alert rules: {e}")
                
                time.sleep(interval_seconds)
        
        self._check_thread = threading.Thread(target=check_loop, daemon=True)
        self._check_thread.start()
        logger.info(f"Started alert checking (interval: {interval_seconds}s)")
    
    def stop_checking(self):
        """Stop checking."""
        self._checking = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        logger.info("Stopped alert checking")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self.alerts.values() if not a.resolved]


class HealthChecker:
    """
    Perform health checks on system and dependencies.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func
    
    def check_database(self) -> bool:
        """Check database connectivity."""
        try:
            # Import here to avoid circular dependency
            from db.database import SessionLocal
            
            db = SessionLocal()
            # Simple query
            db.execute("SELECT 1")
            db.close()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def check_redis(self) -> bool:
        """Check Redis connectivity."""
        try:
            import redis
            from utils.settings import settings
            
            r = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                socket_connect_timeout=5
            )
            r.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def check_disk_space(self, min_free_gb: float = 5.0) -> bool:
        """Check available disk space."""
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024 ** 3)
        return free_gb >= min_free_gb
    
    def check_memory(self, max_percent: float = 95.0) -> bool:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        return memory.percent < max_percent
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.
        
        Returns:
            Health check results
        """
        results = {}
        all_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                healthy = check_func()
                results[name] = {
                    'status': 'healthy' if healthy else 'unhealthy',
                    'timestamp': time.time()
                }
                if not healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': time.time()
                }
                all_healthy = False
        
        overall_status = HealthStatus.HEALTHY if all_healthy else HealthStatus.UNHEALTHY
        
        return {
            'status': overall_status.value,
            'checks': results,
            'timestamp': time.time()
        }


# Singleton instances
_metrics_collector: Optional[MetricsCollector] = None
_system_monitor: Optional[SystemMonitor] = None
_alert_manager: Optional[AlertManager] = None
_health_checker: Optional[HealthChecker] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_system_monitor() -> SystemMonitor:
    """Get or create system monitor."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor(get_metrics_collector())
    return _system_monitor


def get_alert_manager() -> AlertManager:
    """Get or create alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(get_metrics_collector())
    return _alert_manager


def get_health_checker() -> HealthChecker:
    """Get or create health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
        # Register default checks
        _health_checker.register_check('database', _health_checker.check_database)
        _health_checker.register_check('redis', _health_checker.check_redis)
        _health_checker.register_check('disk_space', _health_checker.check_disk_space)
        _health_checker.register_check('memory', _health_checker.check_memory)
    return _health_checker
