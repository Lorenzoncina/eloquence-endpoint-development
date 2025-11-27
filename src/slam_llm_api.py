import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form
from pydantic import BaseModel
import torch
import os
import sys
import logging
import contextlib
import shutil
import uuid
from omegaconf import DictConfig
from typing import Optional
from transformers import GenerationConfig

# --- Dependency Imports ---
try:
    import slam_llm
    import peft
    import whisper
    from omegaconf import OmegaConf
except ImportError as e:
    print(f"\n[ERROR] Library import failed: {e}")
    sys.exit(1)

# --- SLAM-LLM Path Setup ---
SLAM_LLM_EXAMPLE_PATH = "/workspace/SLAM-LLM/examples/asr_librispeech"
if SLAM_LLM_EXAMPLE_PATH not in sys.path:
    sys.path.append(SLAM_LLM_EXAMPLE_PATH)

try:
    from model_factory_local import model_factory
except ImportError as e:
    print(f"\n[ERROR] Failed to import 'model_factory': {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Lifespan: Load Model ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- Starting server and loading SLAM-LLM model... ---")
    app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Paths
    LLM_PATH = os.getenv("LLM_PATH", "/models/eurollm-1.7b")
    SPEECH_ENCODER_PATH = os.getenv("SPEECH_ENCODER_PATH", "/models/large-v3-turbo.pt")
    PROJECTOR_CKPT_PATH = os.getenv("PROJECTOR_CKPT_PATH", "/checkpoints/model.pt")

    # 2. Config (Must match training)
    model_config = DictConfig({
        "llm_name": "eurollm-1.7b",
        "llm_path": LLM_PATH,
        "llm_dim": 2048,
        "encoder_name": "whisper",
        "encoder_projector_ds_rate": 5,
        "encoder_path": SPEECH_ENCODER_PATH,
        "encoder_dim": 1280,
        "encoder_projector": "linear",
        "whisper_decode": False,
        "encoder_path_hf": None,
        "normalize": True,
    })

    train_config = DictConfig({
        "model_name": "asr",
        "freeze_encoder": True,
        "freeze_llm": True,
        "use_peft": True, 
        "enable_fsdp": False,
        "enable_ddp": False,
        "quantization": False,
        "peft_config": {
            "r": 8, 
            "lora_alpha": 32, 
            "lora_dropout": 0.05, 
            "target_modules": ["q_proj", "v_proj"] 
        }
    })
    
    dataset_config = DictConfig({"mel_size": 128})

    try:
        # 3. Load Model
        model, tokenizer = model_factory(
            train_config,
            model_config,
            ckpt_path=PROJECTOR_CKPT_PATH,
        )
        
        # Ensure PAD is EOS to prevent infinite generation
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.to(app.state.device)
        model.eval()

        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.dataset_config = dataset_config
        logger.info("--- Model loaded successfully ---")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    yield
    del app.state.model
    del app.state.tokenizer
    if app.state.device == "cuda":
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class AsrResponse(BaseModel):
    key: str
    transcription: str

# --- API Endpoint (MODIFIED FOR FILE UPLOAD) ---
@app.post("/v1/transcribe", response_model=AsrResponse)
async def create_transcription(
    request: Request,
    key: str = Form(...),
    file: UploadFile = File(...)
):
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device
    
    # Generate a unique temp filename
    temp_filename = f"/tmp/{uuid.uuid4()}_{file.filename}"
    
    try:
        # 1. Save the uploaded file to disk
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if not os.path.exists(temp_filename):
            return AsrResponse(key=key, transcription="<ERROR: File upload failed>")

        # 2. Prepare Prompt
        raw_prompt = "Transcribe speech to text. "
        prompt_text = f"USER: {raw_prompt}\n ASSISTANT:"

        # 3. Audio Encoding (Using the temp file)
        audio_raw = whisper.load_audio(temp_filename)
        audio_raw = whisper.pad_or_trim(audio_raw)
        mel_size = request.app.state.dataset_config.get("mel_size", 128)
        audio_mel = (
            whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
            .permute(1, 0)[None, :, :].to(device)
        )

        with torch.no_grad():
            encoder_outs = model.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1))
            if model.model_config.encoder_projector == "linear":
                encoder_outs = model.encoder_projector(encoder_outs)
            
            # 4. Text Encoding
            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)
            
            if hasattr(model.llm.model, "embed_tokens"):
                inputs_embeds = model.llm.model.embed_tokens(prompt_ids)
            elif hasattr(model.llm.model.model, "embed_tokens"): 
                inputs_embeds = model.llm.model.model.embed_tokens(prompt_ids)
            else:
                inputs_embeds = model.llm.model.model.model.embed_tokens(prompt_ids)

            # 5. Concatenation & Generation
            combined_embeds = torch.cat((encoder_outs, inputs_embeds), dim=1)
            attention_mask = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(device)

            generation_config = GenerationConfig(
                max_new_tokens=256,
                min_new_tokens=1,
                num_beams=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )

            generated_ids = model.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config
            )

            raw_transcription = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            
            # Post-Processing
            if '♀' in raw_transcription:
                raw_transcription = raw_transcription.split('♀')[0].strip()

            return AsrResponse(key=key, transcription=raw_transcription)

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return AsrResponse(key=key, transcription=f"<ERROR: {str(e)}>" )
        
    finally:
        # 6. Cleanup: Remove the temp file to save space
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass

if __name__ == "__main__":
    uvicorn.run("slam_llm_api:app", host="0.0.0.0", port=8080, reload=False)