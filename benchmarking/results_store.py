"""
Bit 9: Results Storage & Persistence
Manages export of benchmark experiments to structured CSV and JSON files in benchmarking/results/
Guarantees non-destructive persistence with unique experiment IDs.
"""

import csv
import json
import os
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .metrics import RunMetrics, BenchmarkSummary


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class ResultsStore:
    """Thread-safe persistence manager for benchmark results."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or RESULTS_DIR
        self._lock = threading.Lock()
        os.makedirs(self.output_dir, exist_ok=True)
        self.index_file = os.path.join(self.output_dir, "experiments_index.json")

    def save_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        runs: List[RunMetrics],
        summaries: Dict[str, BenchmarkSummary],
        task_records: Optional[List[Dict[str, Any]]] = None,
        ablations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Save a complete benchmark experiment to JSON and CSV artifacts."""
        with self._lock:
            timestamp = datetime.now(timezone.utc).isoformat()

            # 1. Save Summary JSON
            summary_path = os.path.join(self.output_dir, f"experiment_{experiment_id}_summary.json")
            payload = {
                "experiment_id": experiment_id,
                "timestamp": timestamp,
                "config": config,
                "summaries": {k: v.to_dict() for k, v in summaries.items()},
                "ablations": ablations or {},
                "total_runs": len(runs),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            # 2. Save Runs CSV
            runs_csv_path = os.path.join(self.output_dir, f"experiment_{experiment_id}_runs.csv")
            if runs:
                run_dicts = [r.to_dict() for r in runs]
                headers = list(run_dicts[0].keys())
                with open(runs_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(run_dicts)

            # 3. Save Tasks CSV (if provided)
            tasks_csv_path = os.path.join(self.output_dir, f"experiment_{experiment_id}_tasks.csv")
            if task_records:
                task_headers = list(task_records[0].keys())
                with open(tasks_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=task_headers)
                    writer.writeheader()
                    writer.writerows(task_records)

            # 4. Update Index JSON
            self._update_index(experiment_id, timestamp, config, len(runs), summary_path)

            return {
                "summary_json": summary_path,
                "runs_csv": runs_csv_path,
                "tasks_csv": tasks_csv_path if task_records else "",
            }

    def _update_index(self, experiment_id: str, timestamp: str, config: Dict[str, Any], total_runs: int, summary_path: str):
        """Append or update the experiment index."""
        index_data = []
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
            except Exception:
                index_data = []

        # Remove existing if same ID
        index_data = [item for item in index_data if item.get("experiment_id") != experiment_id]

        index_data.append({
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "name": config.get("name", "Benchmark"),
            "schedulers": config.get("schedulers", []),
            "num_tasks": config.get("num_tasks", 0),
            "num_repetitions": config.get("num_repetitions", 0),
            "total_runs": total_runs,
            "summary_file": os.path.basename(summary_path),
        })

        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all indexed experiments."""
        if not os.path.exists(self.index_file):
            return []
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment summary by experiment_id."""
        path = os.path.join(self.output_dir, f"experiment_{experiment_id}_summary.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
