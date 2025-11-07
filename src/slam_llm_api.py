import uvicorn
from fastapi import FastAPI
import torch
import os
import sys

print("--- Attempting to import SLAM-LLM and dependencies... ---")

try:
    # --- This is the REAL test ---
    # Can we import the libraries we installed from git in the Dockerfile?
    # These libraries are installed directly into site-packages.
    import slam_llm
    import fairseq
    import peft
    
    print("[SUCCESS] Successfully imported:")
    print(f"  - slam_llm (from {slam_llm.__file__})")
    print(f"  - fairseq (from {fairseq.__file__})")
    print(f"  - peft (from {peft.__file__})")
    
except ImportError as e:
    print(f"\n[ERROR] Failed to import a core library. This indicates a problem.")
    print(f"Details: {e}")
    print("Please check the Dockerfile installation steps for SLAM-LLM and its dependencies.")
    print(f"Current Python path: {sys.path}\n")
    # Raise the error to stop the container if imports fail
    raise e

print("----------------------------------------------------")


app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "SLAM-LLM API server is running.",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/v1/chat/completions")
def test_chat():
    # This is where you will load your trained SLAM-LLM model
    # and add the inference logic.
    return {"message": "This is the SLAM-LLM chat endpoint. Ready for development."}

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Starting Uvicorn server on http://0.0.0.0:8080")
    uvicorn.run(
        "slam_inference_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
         reload_dirs=["/app"],
    )