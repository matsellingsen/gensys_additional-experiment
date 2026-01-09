import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

def save_experiment_results(
    results: Dict[int, Dict[str, Any]],
    filename: str,
    experiment_nr: int = 0,
    description: Optional[str] = None,
    results_dir: str = "results",
):
    """
    Save experiment results to a timestamped JSON file.

    Timestamp format: DDMMYYYYHHMM
    """
    
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    output_dir = results_dir+"/experiment_"+str(experiment_nr)
    full_filename = filename + f".{timestamp}.json"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "experiment": experiment_nr,
        "description": description,
        "timestamp": timestamp,
        "results": {str(k): v for k, v in results.items()},
    }

    save_path = os.path.join(output_dir, full_filename)

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=4)

    print(f"Results saved to {save_path}")



def load_experiment_results(filepath: str) -> Dict[int, Dict[str, Any]]:
    """
    Load experiment results from a JSON file and return the results dict.

    Converts string keys back to integers.
    """
    with open(filepath, "r") as f:
        payload = json.load(f)

    results = {
        int(k): v for k, v in payload["results"].items()
    }

    return results
