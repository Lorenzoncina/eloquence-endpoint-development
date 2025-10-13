import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models.aligned_decoder_lm import SpeechEncoderConnectorLLM
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoTokenizer
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
    audio_path="audio",
    text_path="text",
    model_input_name="input_features",
    prompt_prefix="",
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
    model_inputs = (
        model_inputs.to(model.device) if hasattr(model_inputs, "to") else model_inputs
    )
    # Generate output
    outputs = model.generate(**model_inputs, generation_config=model.generation_config)
    generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return JSONResponse(
        content={
            "choices": [
                {"message": {"role": "assistant", "content": generated_batch[0]}}
            ]
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "openai_woz_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=["/src"],
    )
