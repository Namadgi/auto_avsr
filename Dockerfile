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
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

RUN python3 -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

RUN pip install -U pip setuptools

# For CUDA  install 
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

# RUN git lfs install
RUN mkdir /home/dependencies

# Pip dependencies
COPY requirements.txt /home/model-server/requirements.txt
RUN pip install --no-cache-dir -r /home/model-server/requirements.txt

# Add user for execute the commands 
RUN useradd -m model-server

RUN cd /home/dependencies && \
    git clone https://github.com/hhj1897/face_detection.git && \
    cd /home/dependencies/face_detection && \
    git lfs pull && \
    pip install -e .

RUN cd /home/dependencies && \
    git clone https://github.com/hhj1897/face_alignment.git && \
    cd /home/dependencies/face_alignment && \
    pip install -e . --pre


# Install extra packages
RUN pip install torchserve==0.9.0 torch-model-archiver
RUN pip install pyyaml

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
RUN pip install strsimpy
RUN pip install setuptools==69.5.1

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY config.properties /home/model-server/config.properties

# Create two folder for models 
RUN mkdir /home/model-server/model-store
# Copy all required models and pipelines inside docker 
COPY model_store /home/model-server/model-store

# RUN mkdir /home/model-server/models
# COPY models /home/model-server/models

# RUN mkdir /home/model-server/data
# COPY data /home/model-server/data

# RUN mkdir /home/model-server/model_weights
# COPY model_weights /home/model-server/model_weights

# Giving rights for execute for entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /home/model-server/tmp \
    && chown -R model-server /home/model-server

# RUN mkdir -p /home/model-server/tmp \
#     && chown -R model-server /home/model-server


# GIVING rights to execute 
RUN chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081 8082

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve", "curl"]
