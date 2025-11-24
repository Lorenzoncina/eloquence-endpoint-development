import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import os
import sys
import logging
import contextlib
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
        # CRITICAL: We pass ckpt_path to load the Projector. 
        # If model.pt also contains PEFT weights, `load_state_dict` in model_factory will try to load them.
        model, tokenizer = model_factory(
            train_config,
            model_config,
            ckpt_path=PROJECTOR_CKPT_PATH,
        )
        
        # CRITICAL FIX: Do NOT add <audio> token or resize embeddings. 
        # The training likely relied on standard tokens or placeholders. 
        # Resizing invalidates weights if the checkpoint didn't have this token.
        
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

class AsrRequest(BaseModel):
    key: str
    source: str 
    prompt: Optional[str] = "Transcribe speech to text."

class AsrResponse(BaseModel):
    key: str
    transcription: str

@app.post("/v1/transcribe", response_model=AsrResponse)
async def create_transcription(request: AsrRequest, app_request: Request):
    model = app_request.app.state.model
    tokenizer = app_request.app.state.tokenizer
    device = app_request.app.state.device
    
    # 1. Chat Template Construction
    # Matches SLAM-LLM SpeechDatasetJsonl format EXACTLY
    raw_prompt = "Transcribe speech to text. "
    prompt_text = f"USER: {raw_prompt}\n ASSISTANT:"
    
    if not os.path.exists(request.source):
        return AsrResponse(key=request.key, transcription="<ERROR: File not found>")

    try:
        # --- Audio Encoding ---
        audio_raw = whisper.load_audio(request.source)
        audio_raw = whisper.pad_or_trim(audio_raw)
        mel_size = app_request.app.state.dataset_config.get("mel_size", 128)
        audio_mel = (
            whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
            .permute(1, 0)[None, :, :].to(device)
        )

        with torch.no_grad():
            encoder_outs = model.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1))
            if model.model_config.encoder_projector == "linear":
                encoder_outs = model.encoder_projector(encoder_outs)
            
            # --- Text Encoding ---
            # add_special_tokens=True ensures BOS token is present (e.g. <s>)
            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)
            
            # Embed Text
            if hasattr(model.llm.model, "embed_tokens"):
                inputs_embeds = model.llm.model.embed_tokens(prompt_ids)
            elif hasattr(model.llm.model.model, "embed_tokens"): # Adapters often wrap logic
                inputs_embeds = model.llm.model.model.embed_tokens(prompt_ids)
            else:
                inputs_embeds = model.llm.model.model.model.embed_tokens(prompt_ids)

            # --- Concatenation: [Audio] + [Text] ---
            combined_embeds = torch.cat((encoder_outs, inputs_embeds), dim=1)
            attention_mask = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(device)

            # --- Generation ---
            generation_config = GenerationConfig(
                max_new_tokens=256,
                min_new_tokens=1, # Changed from 2 to 1 to allow short answers
                num_beams=1,      # Greedy decoding is strictly more stable for stopping
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id, # Force PAD to EOS behavior
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )

            generated_ids = model.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config
            )

            raw_transcription = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            
            # --- Post-Processing Failsafe ---
            # If the model still outputs the garbage char '♀', cut it off manually
            if '♀' in raw_transcription:
                raw_transcription = raw_transcription.split('♀')[0].strip()

            return AsrResponse(key=request.key, transcription=raw_transcription)

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return AsrResponse(key=request.key, transcription=f"<ERROR: {str(e)}>" )

if __name__ == "__main__":
    uvicorn.run("slam_llm_api:app", host="0.0.0.0", port=8080, reload=False)