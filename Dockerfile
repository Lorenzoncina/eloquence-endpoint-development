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
# **** THIS STEP IS MODIFIED to pin pip to 23.3.1 ****
RUN python3 -m pip install --upgrade --no-cache-dir pip==23.3.1 && \
    python3 -m pip install --no-cache-dir \
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
        whisperx \
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

# 4. Install the correct torchaudio version for CUDA 11.8
RUN pip install --no-cache-dir torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 5. Set up the /workspace and install SLAM-LLM and its specific deps
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

# 6. Set the final working directory to /app to match your docker-compose.yml
WORKDIR /app