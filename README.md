# 1) Dockerfile (baseline + optional extras)

**Use this as your image definition.**

```docker
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
```

---

# 2) Local development

**Goal:** implement and test your API locally before shipping to MN5.

Start by cloning the base repository:

```bash
git clone https://github.com/MiguelClaramunt/eloquence-endpoint-development.git
cd eloquence-endpoint-development
```

## 2.1) Project layout

```
.
|   .gitignore
|   docker-compose.yml
|   Dockerfile
|   README.md
|
\---src
    |   openai_woz_api.py
    |   whisper_api_python.py
    |   _launch.sh
```

## 2.2) Run locally with Docker Compose (hot‑reload)

Compose drives the dev loop. The `uvicorn` engine in `whisper_api_python.py` is configured as:

```python
uvicorn.run(
    "whisper_api_python:app",
    host="0.0.0.0",
    port=8080,
    reload=True,
    reload_dirs=["/src"],
)
```

This triggers StatReload on any Python change under `/src`. No container restarts.

Run:

```bash
docker compose up
```

Edit any `src/*.py` file. StatReload detects changes and live‑reloads the endpoint.

Smoke tests:

```bash
curl http://localhost:8080/v1/audio/transcriptions -F file=@EloqConv_00.wav
curl http://localhost:8080/v1/audio/transcriptions -F file=@tts_test.mp3 -F model=""
```

## 2.3) API Schema: **Follow OpenAI format** **Follow OpenAI format**

- **Chat Completions**: `POST /v1/chat/completions`
    - Spec: https://platform.openai.com/docs/api-reference/chat/create
    - Input: `model`, `messages`, plus optional decoding params.
    - Output: `choices[].message` and usage counts.
- **Audio Transcriptions**: `POST /v1/audio/transcriptions`
    - Spec: https://platform.openai.com/docs/api-reference/audio/createTranscription
    - Input: multipart file `file`, optional params.
    - Output: transcription text/JSON.
- **vLLM OpenAI-Compatible**: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

**Don’t invent URLs**. `/v2/chat/completions` is invalid. Use only documented endpoints.

**⚠️ Load models at module import time. ⚠️** All heavy initialization and model loading must occur when the Python module is first imported, not within request-handling functions. Avoid calling `from_pretrained()` or any similar model constructor inside a route such as `@app.post`, because every incoming request would reinitialize the model in GPU memory. This behavior leads to GPU memory exhaustion (OOM) and extreme latency due to repeated loading. Placing model loading at module import ensures a single shared model instance is maintained throughout the process lifetime, improving throughput and stability.


## 2.4) Examples

### 2.4.1) Speech-Text WOZ Endpoint (FastAPI + custom model)

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoFeatureExtractor
from pathlib import Path
import sys

from models.aligned_decoder_lm import SpeechEncoderConnectorLLM
from utilities.collators import WOZCollator

# Load model, tokenizer, feature extractor at startup
MODEL_NAME = "pirxus/wavlm-large_olmo1b_lora_r16a16_np_ua_swft"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = SpeechEncoderConnectorLLM.from_pretrained(MODEL_NAME)
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

data_collator = WOZCollator(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    audio_path='audio',
    text_path='text',
    model_input_name='input_features',
    prompt_prefix='',
)

app = FastAPI()

class ChatCompletionRequest(BaseModel):
    audio: str  # path to audio file or base64-encoded audio
    text: str = None  # optional text prompt

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Prepare example for collator
    example = {"audio": request.audio}
    if request.text:
        example["text"] = request.text
    # Prepare model inputs
    model_inputs = data_collator([example])
    model_inputs = model_inputs.to(model.device) if hasattr(model_inputs, 'to') else model_inputs
    # Generate output
    outputs = model.generate(**model_inputs, generation_config=model.generation_config)
    generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return JSONResponse(content={"choices": [{"message": {"role": "assistant", "content": generated_batch[0]}}]})

if __name__ == "__main__":
    uvicorn.run(
        "openai_woz_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=["/src"],
    )

```

### 2.4.2) Example: WhisperX Transcription Endpoint

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisperx, tempfile, json
from pathlib import Path

device = "cuda"
model = whisperx.load_model("large-v2", device, compute_type="float16")
model_a, metadata = whisperx.load_align_model(language_code="es", device=device)
app = FastAPI()

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    diarize_model = whisperx.diarize.DiarizationPipeline(device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    out_jsonl = Path(tempfile.mkdtemp()) / f"{Path(audio_path).stem}.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(json.dumps({"speaker": seg.get("speaker", "unknown"), "text": seg["text"]}, ensure_ascii=False) + "\\n")
    return FileResponse(out_jsonl, media_type="application/jsonl", filename=out_jsonl.name)

if __name__ == "__main__":
    uvicorn.run(
        "whisper_api_python:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=["/src"],
    )

```

---

# Contact

For documentation clarifications or typos, contact: [miguelclaramunt.bsc@gmail.com](mailto:miguelclaramunt.bsc@gmail.com)