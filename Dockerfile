FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
# FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

ENV PYTHONUNBUFFERED TRUE

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    python3-dev \
    python3-distutils \
    python3-venv \
    openjdk-11-jre-headless \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/pip/3.8/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

RUN python3 -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

RUN pip install -U pip setuptools

# For CUDA install 
RUN export USE_CUDA=1
ARG CUDA=1
RUN if [ $CUDA==1 ]; then \ 
        pip install nvgpu; \
    fi

# Extra libraries for opencv
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y  \
    bzip2 \
    g++ \
    git \
    git-lfs \
    graphviz \
    libgl1-mesa-glx \
    libhdf5-dev \
    openmpi-bin \
    wget \
    python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
RUN mkdir -p /home/app
WORKDIR /home/app

# Copy requirements first for better caching
COPY requirements.txt /home/app/requirements.txt
RUN pip install --no-cache-dir -r /home/app/requirements.txt

# Install ffmpeg
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install gdown

# Create dependencies directory and install face detection
RUN mkdir /home/dependencies && \
    cd /home/dependencies && \
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/hhj1897/face_detection.git && \
    cd /home/dependencies/face_detection && \
    gdown https://drive.google.com/uc?id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1 && \
    gdown https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW && \
    gdown https://drive.google.com/uc?id=1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt && \
    mv mobilenet0.25_Final.pth ibug/face_detection/retina_face/weights/ && \
    mv Resnet50_Final.pth      ibug/face_detection/retina_face/weights/ && \
    mv sfd_face.pth            ibug/face_detection/s3fd/weights/s3fd_weights.pth && \
    pip install -e .

# Install face alignment
RUN cd /home/dependencies && \
    git clone https://github.com/hhj1897/face_alignment.git && \
    cd /home/dependencies/face_alignment && \
    pip install -e . --pre

# Copy application code
COPY src/ /home/app/src/
COPY app.py /home/app/
COPY extra_files/ /home/app/extra_files/
COPY docker-entrypoint.sh /home/app/docker-entrypoint.sh
RUN chmod +x /home/app/docker-entrypoint.sh

# Create user for running the application
RUN useradd -m appuser && \
    chown -R appuser:appuser /home/app && \
    chown -R appuser:appuser /home/dependencies

# Create tmp directory for video processing
RUN mkdir -p /home/app/tmp && \
    chown -R appuser:appuser /home/app/tmp

# Switch to non-root user
USER appuser

# Configure OpenTelemetry based on deployment environment
# For GCP Cloud Run (recommended for GCP):
ENV OTEL_ENV=gcp
ENV OTEL_SERVICE_NAME=gpu-service
ENV OTEL_TRACES_EXPORTER=cloud_trace
ENV OTEL_METRICS_EXPORTER=cloud_monitoring

# COPY gcp.json /home/app/gcp.json
ENV GOOGLE_CLOUD_PROJECT=biometry-416410
# ENV GOOGLE_APPLICATION_CREDENTIALS=gcp.json

# Expose port for FastAPI
EXPOSE 8080

# Run the application
ENTRYPOINT ["/home/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
