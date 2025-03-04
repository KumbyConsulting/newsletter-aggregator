import time
import logging
from .config_service import ConfigService
from functools import wraps
from datetime import datetime

# Add Google Cloud Monitoring imports
from google.cloud import monitoring_v3
from google.api import metric_pb2
from google.api import label_pb2
from google.api import monitored_resource_pb2

class MonitoringService:
    """Service for monitoring application performance and health"""
    
    def __init__(self):
        self.config = ConfigService()
        self.client = None
        
        # Initialize Google Cloud Monitoring if enabled
        if self.config.is_gcp_enabled and hasattr(self.config, 'use_cloud_monitoring') and self.config.use_cloud_monitoring:
            try:
                self.client = monitoring_v3.MetricServiceClient()
                self.project_name = f"projects/{self.config.gcp_project_id}"
                logging.info("Google Cloud Monitoring initialized")
            except Exception as e:
                logging.error(f"Failed to initialize Google Cloud Monitoring: {e}")
                self.client = None
    
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
    
    def time_async_function(self, operation_name):
        """Decorator to time an async function and record its latency"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                result = await func(*args, **kwargs)
                end_time = time.time()
                
                latency = end_time - start_time
                self.record_latency(operation_name, latency)
                
                return result
            return wrapper
        return decorator 