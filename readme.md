# MLOps-Challenge
## Run Training Pipeline

(Need to create a virtual environment first with Python version 3.11)
```
pip install -r requirements.txt

bash train.sh
```

Runs the training job, logs the run with MLflow locally, and saves the trained model in artifacts/.

## Build & Run Inference Container
For Building the container: 

docker build -t aibidia-iris:latest .

Running the container:

docker run --rm -p 8000:8000 aibidia-iris:latest


Starts the FastAPI inference service. Logs (latency, drift, request IDs) are written to logs/inference.jsonl.

## Send Test Predictions
```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.1,3.5,1.4,0.2],[6.7,3.1,4.7,1.5]]}'
```

Calls the running container and returns predictions with probabilities.

SEE with the tail command inside docker the logs:
(New Terminal) 

```
docker ps

docker exec -it <container_id> bash

tail -f /app/logs/inference.jsonl
```

## Monitoring & Drift Simulation

Each prediction adds a JSON entry to logs/inference.jsonl
```

Timestamp

Request ID

Latency (ms)

Batch size

Rolling feature means (last ~500 inputs)

Drift proxy (mean-shift score)

Class prediction counts
```

View logs live:

tail -f logs/inference.jsonl

### Drift Proxy Information

Computed from difference between training means and rolling means

Higher values of the drift proxy means input is drifting away from training distribution

## Approach Summary

In this project, I built a compact version but production style ML pipeline around a simple Iris classifier. The process starts with training script (train.py) that logs metrics, parameters, and artifacts using MLflow. GitHub Actions automates this training step in a clean environment, ensuring consistent results and storing the trained model under artifacts/. The model is deployed through a containerized FastAPI application (app.py), which exposes /predict for inference and writes structured logs for monitoring. Each inference request records latency, rolling feature statistics, a drift proxy, and prediction distribution into logs/inference.jsonl, allowing observation of model behavior. Finally, a test script is used to simulate multiple API hits, demonstrating monitoring output and confirming that the entire pipeline i.e, from training to containerized inference, is working properly. For orchestrations, I used a simple GitHub Actions pipeline that, whenever a new change is pushed, initiates new training and uses the latest model artifact for deployment.
