"""
This script is used to store the existing data in redis 
in JSON format which will be avaible locally to use
"""

import redis
import json
import os
from datetime import datetime
import argparse

# Set up
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_TELEMETRY_KEY = "carla:telemetry"
REDIS_METADATA_KEY = "carla:metadata"
REDIS_STATS_KEY = "carla:stats"


class TelemetryExport:
    """The Exporter class for redis"""
    
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB):
        # Create a connection
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
    
    def get_metadata(self):
        metadata_str = self.redis_client.get(REDIS_METADATA_KEY)
        if metadata_str:
            return json.loads(metadata_str)
        return None
    
    def get_stats(self):
        stats_str = self.redis_client.get(REDIS_STATS_KEY)
        if stats_str:
            return json.loads(stats_str)
        return None
    
    def get_total_entries(self):
        return self.redis_client.llen(REDIS_TELEMETRY_KEY)
    
    def get_recent_entries(self, count=10):
        entries = self.redis_client.lrange(REDIS_TELEMETRY_KEY, -count, -1)
        return [json.loads(entry) for entry in entries]
    
    def get_all_entries(self):
        entries = self.redis_client.lrange(REDIS_TELEMETRY_KEY, 0, -1)
        return [json.loads(entry) for entry in entries]
    
    def get_anomalies(self):
        all_entries = self.get_all_entries()
        return [entry for entry in all_entries if entry.get('anomaly', {}).get('detected', False)]
    
    def export_to_json(self, filepath=None):
        # Default ./carla_ws/logs 
        # if no filepath provided
        if filepath is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(".", "carla_ws", "logs")
            os.makedirs(log_dir, exist_ok=True)
            filepath = os.path.join(log_dir, f"telemetry_{timestamp}.json")
        
        metadata = self.get_metadata()
        all_data = self.get_all_entries()
        
        
        export_data = {
            "metadata": metadata,
            "data": all_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_anomalies_to_json(self, filepath=None):
        """This script will only save the anomalies out of all the logs"""
        
        # Default ./carla_ws/logs
        # if no filepath provided
        if filepath is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(".", "carla_ws", "logs")
            os.makedirs(log_dir, exist_ok=True)
            filepath = os.path.join(log_dir, f"anomalies_{timestamp}.json")
        
        metadata = self.get_metadata()
        anomalies = self.get_anomalies()
        
        export_data = {
            "metadata": metadata,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    
    def clear_data(self):
        """Clear all telemetry data"""
        confirm = input("Are you sure you want to clear all telemetry data? (yes/no): ")
        if confirm.lower() == 'yes':
            self.redis_client.delete(REDIS_TELEMETRY_KEY)
            self.redis_client.delete(REDIS_METADATA_KEY)
            self.redis_client.delete(REDIS_STATS_KEY)
        return 
        
    def close(self):
        """Close Redis connection"""
        self.redis_client.close()


def main():
    parser = argparse.ArgumentParser(description="telemetry from Redis")
    parser.add_argument('--export', type=str, metavar='FILE', nargs='?', const='auto', 
                        help='Export all data to JSON file (default: ./carla_ws/logs/telemetry_TIMESTAMP.json)')
    parser.add_argument('--export-anomalies', type=str, metavar='FILE', nargs='?', const='auto',
                        help='Export anomalies to JSON file (default: ./carla_ws/logs/anomalies_TIMESTAMP.json)')
    parser.add_argument('--clear', action='store_true', help='Clear all telemetry data')
    parser.add_argument('--host', type=str, default=REDIS_HOST, help='Redis host')
    parser.add_argument('--port', type=int, default=REDIS_PORT, help='Redis port')
    
    args = parser.parse_args()
    
    # Initialize Exporter Class
    export = TelemetryExport(host=args.host, port=args.port)
    
    try:
        
        if args.export:
            filepath = None if args.export == 'auto' else args.export
            export.export_to_json(filepath)
        
        if args.export_anomalies:
            filepath = None if args.export_anomalies == 'auto' else args.export_anomalies
            export.export_anomalies_to_json(filepath)
        
        if args.clear:
            export.clear_data()
        
    finally:
        export.close()


if __name__ == "__main__":
    main()
