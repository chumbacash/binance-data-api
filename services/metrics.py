import logging
from datetime import datetime

class MetricsTracker:
    def __init__(self):
        self.logger = logging.getLogger("CryptoPredictMetrics")
        self.error_counts = {}
        
    def track_error(self, endpoint: str):
        """Track error metrics for specific endpoints"""
        self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
        self.logger.warning(f"Error occurred in {endpoint} endpoint. Total errors: {self.error_counts[endpoint]}")
        
    def get_metrics(self):
        """Return current metrics snapshot"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error_counts": self.error_counts.copy()
        } 