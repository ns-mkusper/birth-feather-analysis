# Feather Molt Analysis Pipeline

A distributed feather segmentation pipeline for Apple Silicon Mac minis using Celery workers and a Redis-backed task queue.

## Project Structure
- `data/raw/`: Input feather `.jpg` files.
- `data/processed/`: Output segmented feather crops.
- `src/full_run_distributed.py`: Dispatches all image tasks to the cluster.
- `src/celery_tasks.py`: Distributed task definitions.
- `src/feather_processing.py`: Core segmentation and extraction logic.
- `run_cluster.sh`: Bootstraps Redis, Celery workers, Flower, and starts the pipeline.

## Local Setup
1. Run `./setup_env.sh`
2. Activate env: `conda activate feather_env`

## Cluster Run (4 Mac minis)
1. Ensure your SSH key and host IPs in `run_cluster.sh` are correct.
2. Place all feather images in `data/raw/`.
3. Launch cluster + pipeline:
   ```bash
   ./run_cluster.sh
   ```

## Monitoring
- Flower dashboard: `http://<head-ip>:5555`
- Pipeline log on head: `distributed_pipeline.log`
- Worker logs on each node: `celery_worker.log`
