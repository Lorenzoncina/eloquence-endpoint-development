import io
import os
import subprocess

import librosa
import uvicorn
import whisperx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile


device = "cuda"
batch_size = 16
compute_type = "float16"

model = whisperx.load_model("large-v2", device, compute_type=compute_type)

model_a, metadata = whisperx.load_align_model(language_code="es", device=device)

app = FastAPI()


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        audio_bytes = io.BytesIO(content)
        audio, sr = librosa.load(audio_bytes, sr=None, mono=False)

        result = model.transcribe(audio, batch_size=batch_size)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise HTTPException(
                status_code=500, detail="HF_TOKEN not configured in environment"
            )
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )

        diarize_segments = diarize_model(audio)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        segments_out = []
        for segment in result["segments"]:
            segments_out.append(
                {
                    "speaker": segment.get("speaker", "unknown"),
                    "text": segment["text"],
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                }
            )

        return {"segments": segments_out}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"WhisperX failed: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "whisper_api_python:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=["/src"],
    )
