FROM ghcr.io/huggingface/text-generation-inference:3.3.6

# 1. Merged APT dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libavfilter-dev \
        libswscale-dev \
        libswresample-dev \
        pkg-config \
        python3 \
        python3-pip \
        python3.10-dev \
        git \
        build-essential \
        sox \
        libsox-fmt-all \
        libsox-fmt-mp3 \
        libsndfile1-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Merged PIP dependencies (Pinning pip)
RUN python3 -m pip install --upgrade --no-cache-dir pip==23.3.1 && \
    python3 -m pip install --no-cache-dir \
        pydantic \
        starlette \
\
        fastapi \
        sentencepiece \
        torch \
        tslearn \
        h5py \
        uvicorn \
        python-multipart \
        torchcodec \
        numpy \
        accelerate \
        librosa \
\
        soundfile \
        audioread \
        whisperx \
        hf_transfer \
        # Added from SLAM-LLM
        packaging \
        editdistance \
        gpustat \
        wandb \
        einops \
        debugpy \
        tqdm \
        matplotlib \
        scipy \
        pandas \
        torchaudio==2.1.0

# 3. Add SLAM-LLM git clone and install steps
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

# 4. Set the working directory back to /app
WORKDIR /app