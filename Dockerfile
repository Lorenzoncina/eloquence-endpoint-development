FROM ghcr.io/huggingface/text-generation-inference:3.3.6

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade --no-cache-dir pip && \
    python3 -m pip install --no-cache-dir \
        pydantic \
        starlette \
        transformers==4.48.2 \
        fastapi \
        sentencepiece \
        torch \
        tslearn \
        peft \
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
        hf_transfer
