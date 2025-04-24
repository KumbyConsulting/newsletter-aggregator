"""Enhanced monitoring service with Prometheus metrics and Cloud Monitoring integration."""
import time
import logging
from functools import wraps
from datetime import datetime
from typing import Dict, Any, Optional
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from google.cloud import monitoring_v3
from google.api import metric_pb2, label_pb2
import threading
from .config_service import ConfigService

class MonitoringService:
    """Service for monitoring application performance and health"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MonitoringService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.config = ConfigService()
        
        # Initialize Prometheus metrics
        self.request_latency = Histogram(
            'request_duration_seconds',
            'Request latency in seconds',
            ['endpoint', 'method', 'status_code']
        )
        
        self.request_count = Counter(
            'request_total',
            'Total request count',
            ['endpoint', 'method', 'status_code']
        )
        
        self.queue_depth = Gauge(
            'task_queue_depth',
            'Current task queue depth'
        )
        
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['name']
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Initialize Google Cloud Monitoring if enabled
        self.client = None
        if self.config.use_cloud_logging and self.config.gcp_project_id:
            try:
                self.client = monitoring_v3.MetricServiceClient()
                self.project_name = f"projects/{self.config.gcp_project_id}"
            except Exception as e:
                logging.error(f"Failed to initialize Cloud Monitoring: {e}")
        
        # Start Prometheus metrics server
        try:
            start_http_server(8000)
            logging.info("Prometheus metrics server started on port 8000")
        except Exception as e:
            logging.error(f"Failed to start Prometheus metrics server: {e}")
        
        # Start metrics collection thread
        self._start_metrics_collection()
        self._initialized = True
    
    def _start_metrics_collection(self):
        """Start background thread for collecting system metrics"""
        def collect_metrics():
            while True:
                try:
                    # Update system metrics
                    self.memory_usage.set(psutil.Process().memory_info().rss)
                    self.cpu_usage.set(psutil.Process().cpu_percent())
                    
                    # Sleep for 15 seconds
                    time.sleep(15)
                except Exception as e:
                    logging.error(f"Error collecting metrics: {e}")
        
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()
    
    def record_request_metric(self, endpoint: str, method: str, duration: float, status_code: int):
        """Record request metrics"""
        try:
            # Record in Prometheus
            self.request_latency.labels(endpoint=endpoint, method=method, status_code=status_code).observe(duration)
            self.request_count.labels(endpoint=endpoint, method=method, status_code=status_code).inc()
            
            # Record in Cloud Monitoring if enabled
            if self.client:
                self._record_cloud_metric(
                    'custom.googleapis.com/newsletter/request_duration',
                    duration,
                    {'endpoint': endpoint, 'method': method, 'status_code': str(status_code)}
                )
        except Exception as e:
            logging.error(f"Error recording request metric: {e}")
    
    def update_queue_depth(self, depth: int):
        """Update task queue depth metric"""
        try:
            self.queue_depth.set(depth)
            
            if self.client:
                self._record_cloud_metric(
                    'custom.googleapis.com/newsletter/queue_depth',
                    depth
                )
                
            if depth > 5:  # Alert on high queue depth
                logging.warning(f"High task queue depth: {depth}")
        except Exception as e:
            logging.error(f"Error updating queue depth: {e}")
    
    def update_circuit_breaker(self, name: str, state: str):
        """Update circuit breaker state metric"""
        try:
            state_value = {'closed': 0, 'half-open': 1, 'open': 2}.get(state, 0)
            self.circuit_breaker_state.labels(name=name).set(state_value)
            
            if self.client:
                self._record_cloud_metric(
                    'custom.googleapis.com/newsletter/circuit_breaker_state',
                    state_value,
                    {'name': name}
                )
        except Exception as e:
            logging.error(f"Error updating circuit breaker state: {e}")
    
    def _record_cloud_metric(self, metric_type: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record metric in Google Cloud Monitoring"""
        try:
            if not self.client:
                return
                
            series = monitoring_v3.TimeSeries()
            series.metric.type = metric_type
            
            if labels:
                series.metric.labels.update(labels)
            
            series.resource.type = 'cloud_run_revision'
            series.resource.labels.update({
                'project_id': self.config.gcp_project_id,
                'service_name': 'newsletter-aggregator',
                'location': self.config.gcp_region
            })
            
            point = series.points.add()
            point.value.double_value = value
            point.interval.end_time.seconds = int(time.time())
            
            self.client.create_time_series(
                request={
                    "name": self.project_name,
                    "time_series": [series]
                }
            )
        except Exception as e:
            logging.error(f"Error recording Cloud Monitoring metric: {e}")
    
    def time_async_function(self, name: str):
        """Decorator to time async functions"""
        def decorator(func):
            @wraps(func)
            async def wrapped(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.record_request_metric(name, 'ASYNC', duration, 200)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.record_request_metric(name, 'ASYNC', duration, 500)
                    raise
            return wrapped
        return decorator
    
    def record_latency(self, operation_name, latency_seconds):
        """Record operation latency"""
        if self.client:
            try:
                # Create a custom metric descriptor for latency
                metric_type = f"custom.googleapis.com/newsletter_aggregator/{operation_name}_latency"
                
                series = monitoring_v3.TimeSeries()
                series.metric.type = metric_type
                series.metric.labels["operation"] = operation_name
                
                # Create a data point
                point = series.points.add()
                point.value.double_value = latency_seconds
                now = datetime.utcnow()
                point.interval.end_time.seconds = int(now.timestamp())
                
                # Create a resource
                series.resource.type = "global"
                
                # Write the time series
                self.client.create_time_series(
                    name=self.project_name,
                    time_series=[series]
                )
                
                logging.debug(f"Recorded latency for {operation_name}: {latency_seconds}s")
                
            except Exception as e:
                logging.error(f"Failed to record latency in Cloud Monitoring: {e}")
    
    def record_count(self, metric_name, count=1, labels=None):
        """Record a count metric"""
        if self.client:
            try:
                # Create a custom metric descriptor for counts
                metric_type = f"custom.googleapis.com/newsletter_aggregator/{metric_name}"
                
                series = monitoring_v3.TimeSeries()
                series.metric.type = metric_type
                
                # Add labels if provided
                if labels:
                    for key, value in labels.items():
                        series.metric.labels[key] = value
                
                # Create a data point
                point = series.points.add()
                point.value.int64_value = count
                now = datetime.utcnow()
                point.interval.end_time.seconds = int(now.timestamp())
                
                # Create a resource
                series.resource.type = "global"
                
                # Write the time series
                self.client.create_time_series(
                    name=self.project_name,
                    time_series=[series]
                )
                
                logging.debug(f"Recorded count for {metric_name}: {count}")
                
            except Exception as e:
                logging.error(f"Failed to record count in Cloud Monitoring: {e}")
    
    def time_function(self, operation_name):
        """Decorator to time a function and record its latency"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                latency = end_time - start_time
                self.record_latency(operation_name, latency)
                
                return result
            return wrapper
        return decorator
    
    def record_request_duration(self, endpoint, duration, status_code):
        """Record duration of a request by endpoint"""
        # Log the request duration
        logging.debug(f"Request to {endpoint} took {duration:.3f}s (status: {status_code})")
        
        # Record using cloud monitoring if enabled
        if self.client:
            try:
                labels = {
                    "endpoint": endpoint,
                    "status": str(status_code)
                }
                
                # Record as latency
                self.record_latency(f"endpoint_{endpoint}", duration)
                
                # Also record request count by status code
                status_category = str(status_code)[0] + "xx"  # Convert e.g. 404 to "4xx"
                self.record_count("requests", 1, {
                    "endpoint": endpoint,
                    "status": status_category
                })
                
            except Exception as e:
                logging.error(f"Failed to record request duration: {e}") 