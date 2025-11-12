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
# Ensure these are installed in your environment
try:
    import slam_llm
    #import fairseq
    import peft
    import whisper  # Added import for audio processing
    from omegaconf import OmegaConf
except ImportError as e:
    print(f"\n[ERROR] Failed to import a library. Did you install all requirements?")
    print(f"Details: {e}")
    print("Please ensure 'openai-whisper' and 'omegaconf' are installed.")
    sys.exit(1)

# --- SLAM-LLM Example Code Import ---
# Point to the absolute path defined in the Dockerfile
SLAM_LLM_EXAMPLE_PATH = "/workspace/SLAM-LLM/examples/asr_librispeech"
if SLAM_LLM_EXAMPLE_PATH not in sys.path:
    sys.path.append(SLAM_LLM_EXAMPLE_PATH)

# --- SLAM-LLM Example Code Import ---
try:
    # Import from our new, locally-copied file
    from model_factory_local import model_factory
except ImportError as e:
    print(f"\n[ERROR] Failed to import 'model_factory' from model_factory_local.py")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading via Lifespan Context ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the SLAM-LLM model on startup and stores it in the app state.
    """
    logger.info("--- Starting server and loading SLAM-LLM model... ---")

    # 1. Set Device
    app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {app.state.device}")

    # 2. Get Model Paths from Environment Variables
    LLM_PATH = os.getenv("LLM_PATH", "/nfs/maziyang.mzy/models/vicuna-7b-v1.5")
    SPEECH_ENCODER_PATH = os.getenv("SPEECH_ENCODER_PATH", "/nfs/maziyang.mzy/models/Whisper/large-v3.pt")
    PROJECTOR_CKPT_PATH = os.getenv("PROJECTOR_CKPT_PATH", "/root/tmp/vicuna-7b-v1.5-librispeech-linear-steplrwarmupkeep1e-4-whisper-largev3-20240426/asr_epoch_1_step_1000/model.pt")
    

    logger.info(f"LLM Path: {LLM_PATH}")
    logger.info(f"Encoder Path: {SPEECH_ENCODER_PATH}")
    logger.info(f"Projector Ckpt Path: {PROJECTOR_CKPT_PATH}")
   

    # 3. Define Model Configuration (from decode...sh)
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
            "r": 8,             # From your finetune parameters
            "lora_alpha": 32,    # Default from SLAM-LLM
            "lora_dropout": 0.05,  # Default from SLAM-LLM
            "target_modules": ["q_proj", "v_proj"] # Default from SLAM-LLM
        }
    })
    
    # This config is needed by the model's inference method
    dataset_config = DictConfig({
        "mel_size": 128
    })

    # 4. Load the Model
    try:
        model, tokenizer = model_factory(
            train_config,
            model_config,
            ckpt_path=PROJECTOR_CKPT_PATH,
            
        )

        logger.info("Adding <audio> special token to tokenizer.")
        AUDIO_TOKEN = "<audio>"
        tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_TOKEN]})
        model.llm.resize_token_embeddings(len(tokenizer))
        
        model.to(app.state.device)
        model.eval()

        # Store model, tokenizer, and config in app state
        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.model_config = model_config
        app.state.dataset_config = dataset_config

        logger.info("--- Model loaded successfully ---")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    yield

    # --- Cleanup (optional) ---
    logger.info("--- Shutting down and clearing model... ---")
    del app.state.model
    del app.state.tokenizer
    if app.state.device == "cuda":
        torch.cuda.empty_cache()


# --- FastAPI App Definition ---
app = FastAPI(lifespan=lifespan)

class AsrRequest(BaseModel):
    key: str
    source: str # /path/to/audio/file on the server
    prompt: Optional[str] = "Transcribe the following audio."

class AsrResponse(BaseModel):
    key: str
    transcription: str


@app.get("/test")
def read_root():
    """Test endpoint to check if the server is running."""
    return {
        "message": "SLAM-LLM API server is running.",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/v1/transcribe", response_model=AsrResponse)
async def create_transcription(request: AsrRequest, app_request: Request):
    """
    Transcribes an audio file from a given path on the server.
    
    [DEBUG VERSION - PATH A]
    This version reverts to the original manual torch.cat logic
    to debug the audio features themselves.
    """
    model = app_request.app.state.model
    tokenizer = app_request.app.state.tokenizer
    device = app_request.app.state.device
    logger = logging.getLogger(__name__)

    wav_path = request.source
    prompt_text = "Transcribe speech to text. " # Note: using the one from your original file
    
    logger.info(f"--- [DEBUG] Received request for key: {request.key} ---")
    logger.info(f"[DEBUG] Audio file path: {wav_path}")

    if not os.path.exists(wav_path):
        logger.warning(f"[!! ERROR !!] Audio file not found: {wav_path}")
        return AsrResponse(key=request.key, transcription="<ERROR: Audio file not found on server>")

    # --- Inference Logic (Manual Concat) ---
    try:
        # 1. Load and process audio
        logger.info("[DEBUG] Loading audio...")
        audio_raw = whisper.load_audio(wav_path)
        audio_raw = whisper.pad_or_trim(audio_raw)
        mel_size = app_request.app.state.dataset_config.get("mel_size", 128)
        audio_mel = (
            whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
            .permute(1, 0)[None, :, :]
            .to(device)
        )
        logger.info(f"[DEBUG] Audio mel tensor shape: {audio_mel.shape} (batch, frames, mels)")

        with torch.no_grad():
            # 2. Get audio embeddings
            logger.info("[DEBUG] Running audio through model.encoder...")
            # We must permute to (batch, mels, frames) for Whisper encoder
            encoder_outs = model.encoder.extract_variable_length_features(
                audio_mel.permute(0, 2, 1) 
            )
            logger.info(f"[DEBUG] Raw encoder_outs shape: {encoder_outs.shape}")
            logger.info(f"[DEBUG] Raw encoder_outs mean: {encoder_outs.mean().item()}")
            logger.info(f"[DEBUG] Raw encoder_outs std: {encoder_outs.std().item()}")

            # 3. Apply projector
            logger.info("[DEBUG] Running audio through model.encoder_projector...")
            if model.model_config.encoder_projector == "linear":
                encoder_outs = model.encoder_projector(encoder_outs)
            elif model.model_config.encoder_projector == "q-former":
                audio_mel_post_mask = torch.ones(
                    encoder_outs.size()[:-1], dtype=torch.long
                ).to(encoder_outs.device)
                encoder_outs = model.encoder_projector(encoder_outs, audio_mel_post_mask)
            
            logger.info(f"[DEBUG] Projected encoder_outs shape: {encoder_outs.shape}")
            logger.info(f"[DEBUG] Projected encoder_outs mean: {encoder_outs.mean().item()}")
            logger.info(f"[DEBUG] Projected encoder_outs std: {encoder_outs.std().item()}")

            if torch.isnan(encoder_outs).any() or torch.isinf(encoder_outs).any():
                logger.error("[!! CRITICAL !!] Encoder outputs contain NaN or Inf!")
                return AsrResponse(key=request.key, transcription="<ERROR: Bad audio features>")

            # 4. Prepare text prompt (NO <audio> token here)
            formatted_prompt = prompt_text
            prompt_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
            logger.info(f"[DEBUG] Text prompt IDs shape: {prompt_ids.shape}")

            # 5. Get text embeddings
            if hasattr(model.llm.model, "embed_tokens"):
                inputs_embeds = model.llm.model.embed_tokens(prompt_ids)
            elif hasattr(model.llm.model.model, "embed_tokens"):
                inputs_embeds = model.llm.model.model.embed_tokens(prompt_ids)
            else:
                inputs_embeds = model.llm.model.model.model.embed_tokens(prompt_ids)
            logger.info(f"[DEBUG] Text inputs_embeds shape: {inputs_embeds.shape}")

            # 6. Combine embeddings [TEXT, AUDIO]
            combined_embeds = torch.cat((inputs_embeds, encoder_outs), dim=1)
            attention_mask = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(
                device
            )
            logger.info(f"[DEBUG] Combined embeds shape: {combined_embeds.shape}")
            logger.info(f"[DEBUG] Combined attention mask shape: {attention_mask.shape}")
            
            # 7. Generate
            generation_config = GenerationConfig(
                max_new_tokens=256,
                min_new_tokens=2,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

            logger.info("[DEBUG] Calling model.llm.generate() with combined embeds...")
            generated_ids = model.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
            logger.info(f"[DEBUG] Generation complete. Output shape: {generated_ids.shape}")

            # 8. Decode
            raw_transcription = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            ).strip()
            
            logger.info(f"[DEBUG] Raw decoded text (before cleanup): '{raw_transcription}'")

            # Clean up the output, as it might still include the prompt
            if raw_transcription.startswith(prompt_text):
                transcription = raw_transcription[len(prompt_text):].strip()
            else:
                transcription = raw_transcription
            
            logger.info(f"[DEBUG] Final transcription: '{transcription}'")

        logger.info(f"--- [DEBUG] Transcription complete for key: {request.key} ---")
        return AsrResponse(key=request.key, transcription=transcription)

    except Exception as e:
        logger.error(f"Inference failed for key {request.key}: {e}", exc_info=True)
        return AsrResponse(key=request.key, transcription=f"<ERROR: {str(e)}>")

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Starting Uvicorn server on http://0.0.0.0:8080")
    
    # Note: `reload=True` is great for development but should be `False` in production.
    # `reload_dirs` is also for development.
    uvicorn.run(
        "slam_llm_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=["/app"] 
    )