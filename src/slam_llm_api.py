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
    PEFT_CKPT_PATH = os.getenv("PEFT_CKPT_PATH", None) # Support for PEFT checkpoints

    logger.info(f"LLM Path: {LLM_PATH}")
    logger.info(f"Encoder Path: {SPEECH_ENCODER_PATH}")
    logger.info(f"Projector Ckpt Path: {PROJECTOR_CKPT_PATH}")
    logger.info(f"PEFT Ckpt Path: {PEFT_CKPT_PATH}")

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
        "encoder_path_hf": None
    })

    train_config = DictConfig({
        "model_name": "asr",
        "freeze_encoder": True,
        "freeze_llm": True,
        "use_peft": PEFT_CKPT_PATH is not None,
        "enable_fsdp": False, 
        "enable_ddp": False,
        "quantization": False,
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
            peft_ckpt=PEFT_CKPT_PATH
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
    
    This version uses the [TEXT, AUDIO] embedding order, which is
    the last remaining combination to test.
    """
    model = app_request.app.state.model
    tokenizer = app_request.app.state.tokenizer
    device = app_request.app.state.device
    
    wav_path = request.source
    
    # --- FIX 1: Use the correct raw prompt for EuroLLM ---
    prompt_text = "Transcribe speech to text. "

    if not os.path.exists(wav_path):
        return AsrResponse(key=request.key, transcription="<ERROR: Audio file not found on server>")

    # --- Inference Logic ---
    try:
        # 1. Load and process audio
        audio_raw = whisper.load_audio(wav_path)
        audio_raw = whisper.pad_or_trim(audio_raw)
        mel_size = app_request.app.state.dataset_config.get("mel_size", 128)
        audio_mel = (
            whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
            .permute(1, 0)[None, :, :]
            .to(device)
        )

        with torch.no_grad():
            # 2. Get audio embeddings
            encoder_outs = model.encoder.extract_variable_length_features(
                audio_mel.permute(0, 2, 1)
            )

            # 3. Apply projector
            if model.model_config.encoder_projector == "linear":
                encoder_outs = model.encoder_projector(encoder_outs)
            elif model.model_config.encoder_projector == "q-former":
                audio_mel_post_mask = torch.ones(
                    encoder_outs.size()[:-1], dtype=torch.long
                ).to(encoder_outs.device)
                encoder_outs = model.encoder_projector(encoder_outs, audio_mel_post_mask)
            
            # 4. Prepare text prompt
            formatted_prompt = prompt_text
            prompt_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)

            # 5. Get text embeddings
            if hasattr(model.llm.model, "embed_tokens"):
                inputs_embeds = model.llm.model.embed_tokens(prompt_ids)
            elif hasattr(model.llm.model.model, "embed_tokens"):
                inputs_embeds = model.llm.model.model.embed_tokens(prompt_ids)
            else:
                inputs_embeds = model.llm.model.model.model.embed_tokens(prompt_ids)

            # --- FIX 2: Reverse concatenation order to [TEXT, AUDIO] ---
            combined_embeds = torch.cat((inputs_embeds, encoder_outs), dim=1)
            attention_mask = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(
                device
            )
            
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

            logger.info(f"[DEBUG] Manually Combined Embeds Shape: {combined_embeds.shape}")

            generated_ids = model.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
            
            logger.info(f"[DEBUG] Generated IDs Shape: {generated_ids.shape}")

            # 8. Decode
            generated_text_ids = generated_ids[0]
            transcription = tokenizer.decode(
                generated_text_ids, skip_special_tokens=True
            ).strip()
            
            # Clean up the output, as it might still include the prompt
            if transcription.startswith(prompt_text):
                transcription = transcription[len(prompt_text):].strip()
            
            logger.info(f"[DEBUG] Final Transcription: {transcription}")

        logger.info(f"Transcription complete for key: {request.key}")
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