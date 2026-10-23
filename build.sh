export GOOGLE_APPLICATION_CREDENTIALS_JSON=$(cat gcp.json) && \
docker build -t vsr . && \
docker run --rm --gpus all -p 8080:8080 \
  -e GOOGLE_APPLICATION_CREDENTIALS_JSON="$GOOGLE_APPLICATION_CREDENTIALS_JSON" \
  --name vsr vsr