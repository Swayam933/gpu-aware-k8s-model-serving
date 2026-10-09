"""
GPU-aware model serving app. Detects CUDA at startup and reports it via
/healthz — the same code path runs on CPU (for local dev without a GPU) or
GPU (in the target cluster), which is what makes this deployable and
testable without needing a GPU on hand for every step.
"""
import os
import time

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_validator

app = FastAPI(title="gpu-model-serving")

MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")
_model = None
_tokenizer = None
_device = "cpu"


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    max_new_tokens: int = 50

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("prompt must not be blank")
        return cleaned

    @field_validator("max_new_tokens")
    @classmethod
    def validate_max_new_tokens(cls, value: int) -> int:
        if value <= 0 or value > 256:
            raise ValueError("max_new_tokens must be between 1 and 256")
        return value


def _load_model():
    """Load the model once and reuse it for all requests."""
    global _model, _tokenizer, _device
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(_device)


def _ensure_model_loaded():
    if _model is None or _tokenizer is None:
        _load_model()


@app.on_event("startup")
def startup():
    try:
        _load_model()
    except Exception:
        app.state.model_load_error = True
    else:
        app.state.model_load_error = False


@app.get("/")
def root():
    return {"service": "gpu-model-serving", "status": "ok"}


@app.get("/health")
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "device": _device,
        "model_loaded": _model is not None,
    }


@app.get("/ready")
@app.get("/readyz")
def readyz():
    return {
        "status": "ready" if _model is not None else "loading",
        "device": _device,
    }


@app.get("/device")
def device_info():
    """Reports whether this pod actually landed on a GPU node and is using it."""
    import torch

    return {
        "device": _device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    import torch

    try:
        _ensure_model_loaded()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"model failed to load: {exc}",
        ) from exc

    start = time.perf_counter()
    inputs = _tokenizer(req.prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        output = _model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    text = _tokenizer.decode(output[0], skip_special_tokens=True)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "text": text,
        "device": _device,
        "latency_ms": round(latency_ms, 2),
    }
