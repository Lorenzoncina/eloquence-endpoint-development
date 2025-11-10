# 1. Use the SLAM-LLM base image, which has the correct CUDA/PyTorch environment
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

USER root
ARG DEBIAN_FRONTEND=noninteractive

# 2. Install apt dependencies from SLAM-LLM (which includes ffmpeg, git, etc.)
RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim ninja-build \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 3. Install merged PIP dependencies
# **** THIS STEP IS MODIFIED ****
# - Pin pip to 23.3.1
# - Pin torch==2.1.0 and torchaudio==2.1.0 to MATCH the base image's CUDA 11.8
# - Removed 'whisperx' as it conflicts with torch and is not used in this service
RUN python3 -m pip install --upgrade --no-cache-dir pip==23.3.1 && \
    python3 -m pip install --no-cache-dir \
        torch==2.1.0 \
        torchaudio==2.1.0 \
        pydantic \
        starlette \
        fastapi \
        sentencepiece \
        tslearn \
        h5py \
        uvicorn \
        python-multipart \
        torchcodec \
        numpy \
        accelerate \
        librosa \
        soundfile \
        audioread \
        hf_transfer \
        packaging \
        editdistance \
        gpustat \
        wandb \
        einops \
        debugpy \
        tqdm \
        matplotlib \
        scipy \
        pandas

# 4. Set up the /workspace and install SLAM-LLM and its specific deps
WORKDIR /workspace

RUN git clone https://github.com/huggingface/transformers.git \
    && cd transformers \
    && git checkout tags/v4.35.2 \
    && pip install --no-cache-dir -e .

RUN git clone https://github.com/huggingface/peft.git \
    && cd peft \
    && git checkout tags/v0.6.0 \
    && pip install --no-cache-dir -e .

RUN git clone https://github.com/pytorch/fairseq \
    && cd fairseq \
    && pip install --no-cache-dir --editable ./

RUN git clone https://github.com/ddlBoJack/SLAM-LLM.git \
    && cd SLAM-LLM \
    && pip install --no-cache-dir -e .

# 5. Set the final working directory to /app to match your docker-compose.yml
WORKDIR /app