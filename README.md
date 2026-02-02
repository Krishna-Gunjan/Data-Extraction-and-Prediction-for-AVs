# Data Extraction and Anomaly Detection for Autonomous Vehicles 


**Data-Extraction-Using-Carla** is a small collection of scripts that demonstrates how to extract telemetry from the CARLA simulator, train an autoencoder-based anomaly detector from collected logs, run live anomaly detection and export recorded telemetry for analysis.

---

## Project Overview

- Collects vehicle telemetry from CARLA (position, velocity, IMU, GNSS, controls and many more).
- Trains a simple autoencoder on normal driving telemetry to learn typical behavior.
- Detects anomalies in live telemetry using reconstruction error and a tunable threshold.
- Logs telemetry and statistics to Redis for live monitoring and post-processing.
- Provides utilities to export telemetry and anomalies to JSON files.

---

## Features

- Training pipeline: `autoencoder.py` (loads `sample_dataset.json` by default (changeable), trains, saves model and scaler).
- Live detection: `live_redis.py` (reads sensors from CARLA, uses trained model to classify anomalies and logs to Redis).
- Export tool: `redis_query.py` (export all telemetry or anomalies to `./carla_ws/logs/`).

---

## ⚙️ Quickstart

1. Clone the repository:

```bash
git clone https://github.com/<user>/Data-Extraction-and-Prediction-for-AVs.git
cd Data-Extraction-and-Prediction-for-AVs
```

2. Start CARLA server (follow CARLA instructions) and start Redis:

3. (Optional) Create a virtual environment and install dependencies:

4. Train the autoencoder model (this reads `sample_dataset.json`):

```bash
python autoencoder.py

# or run test evaluation after training
python autoencoder.py 1
```

Model and scaler will be saved to `./model/autoencoder.keras` and `./model/scaler.pkl`.

5. Run live anomaly detection (CARLA server must be running):

```bash
python live_redis.py
```

- This script will connect to CARLA, spawn a vehicle, attach IMU/GNSS, load the model and scaler and push telemetry to Redis under the keys:
  - `carla:telemetry` (list of telemetry entries)
  - `carla:metadata` (session metadata)
  - `carla:stats` (summary statistics)

6. Export telemetry and anomalies:

```bash
# export all telemetry to a timestamped file
python redis_query.py --export

# export only anomalies
python redis_query.py --export-anomalies

# clear telemetry 
python redis_query.py --clear
```

Exports default to `./carla_ws/logs/` with timestamped filenames.

---

## Configuration & Tuning

- `ANOMALY_THRESHOLD` in `live_redis.py` controls the sensitivity of live detection (default `0.5`). Tune it according to your training data or use the threshold reported by `autoencoder.py` (95th percentile of reconstruction error) as a starting point.
- Modify the autoencoder architecture or the training hyperparameters in `autoencoder.py` to suit larger or more complex datasets (epochs, batch_size, layer sizes).
- Change vehicle or spawn point in `live_redis.py` (blueprint selection & spawn point index).

---

## Data Format

- `sample_dataset.json` structure:

```json
{
  "metadata": { "vehicle": "Tesla Model 3", "map": "...", "start_time": "..." },
  "data": [
    { "RunTime_s": 0.0, "Roll_deg": 0.0, "Pitch_deg": 0.0, ..., "Anomaly": false, "Reconstruction_Error": 0.000 }
  ]
}
```


## Troubleshooting

- Connection error to CARLA: Ensure the CARLA simulator is running and accessible at `localhost:2000`.
- Redis connection refused: Check Redis is running and reachable at `localhost:6379`.
- GPU / TensorFlow issues: If GPU drivers are not available, install CPU-only TensorFlow or use a machine with CUDA/nvidia drivers configured.


## License

This project is provided under the repository license (see `LICENSE`).

---
