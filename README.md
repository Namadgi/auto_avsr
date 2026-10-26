# Visual Speech Recognition (VSR)
FastAPI service that performs visual speech recognition on uploaded videos or Google Cloud Storage objects, with OpenTelemetry metrics and tracing for GCP.

## Prerequisites
- GPU-capable host with Docker (recommended) and the NVIDIA runtime installed.
- If running without Docker: Python 3.8+, CUDA-enabled PyTorch, FFmpeg, and system git/git-lfs.

The model weights used by the service are already committed at `src/model_weights/vsr.pth`.

## Quick start (Docker, recommended)
```bash
git clone https://github.com/Namadgi/auto_avsr.git
cd auto_avsr

# Optional: provide GCP credentials if you need to access GCS
export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat gcp.json)"   # or set GCP_SA_KEY / GCP_SA_KEY_B64

docker build -t vsr .
docker run --rm --gpus all -p 8080:8080 \
  -e GOOGLE_APPLICATION_CREDENTIALS_JSON="$GOOGLE_APPLICATION_CREDENTIALS_JSON" \
  vsr
```
`docker-entrypoint.sh` materializes the credentials file from one of `GOOGLE_APPLICATION_CREDENTIALS_JSON`, `GCP_SA_KEY`, or `GCP_SA_KEY_B64`. You can also set `PROJECT_ID` to override `GOOGLE_CLOUD_PROJECT`.

## Local development (without Docker)
Only use this if you cannot run Docker; the container already installs everything.
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install FFmpeg via your OS package manager

# Install face detection (with weights) and face alignment
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/hhj1897/face_detection.git
cd face_detection
pip install gdown
gdown https://drive.google.com/uc?id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1
gdown https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW
gdown https://drive.google.com/uc?id=1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt
mv mobilenet0.25_Final.pth ibug/face_detection/retina_face/weights/
mv Resnet50_Final.pth      ibug/face_detection/retina_face/weights/
mv sfd_face.pth            ibug/face_detection/s3fd/weights/s3fd_weights.pth
pip install -e .
cd ..

git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e . --pre
cd ..

uvicorn app:app --host 0.0.0.0 --port 8080
```

## API
- `GET /health` – simple health check.
- `POST /upload` – multipart form upload with `file` field (video). Returns recognition result.
- `POST /predictions/vsr` – JSON body for processing a video stored in GCS:
```json
{
  "instances": [{
    "token": "ya29.c... (OAuth2 bearer token)",
    "bucket_name": "my-bucket",
    "object_name": "path/to/video.mp4",
    "phrase": "HELLO WORLD"
  }]
}
```

Example responses:
```json
{
  "code": 0,
  "description": "Successful check",
  "result": "HELLO WORLD",
  "score": 98.5
}
```
`code` values: `0` phrase matches, `1` phrase does not match, `2` no face detected.

## Example requests
- Upload a local file:
```bash
curl -X POST "http://localhost:8080/upload" \
  -F "file=@data/test_movie.MOV"
```
- Process a GCS object:
```bash
curl -X POST "http://localhost:8080/predictions/vsr" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "token": "ya29.c....",
      "bucket_name": "my-bucket",
      "object_name": "videos/sample.mp4",
      "phrase": "HELLO WORLD"
    }]
  }'
```
