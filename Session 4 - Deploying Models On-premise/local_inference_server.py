
"""
Local Inference Server
Wraps Ollama's API with logging, rate limiting, and consistent response formatting.
This is the foundation of on-premise model deployment.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import defaultdict
import requests
import json
import time
import logging

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:1b"
RATE_LIMIT_REQUESTS = 10      # Max requests per window
RATE_LIMIT_WINDOW_SECONDS = 60  # Window size in seconds

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- FastAPI app ---
app = FastAPI(
    title="Local Inference Server",
    description="A simple on-premise model serving API wrapping Ollama",
    version="1.0.0"
)

# --- Rate limiter (simple in-memory implementation) ---
request_counts = defaultdict(list)

def check_rate_limit(client_ip: str) -> bool:
    """Check if a client has exceeded the rate limit."""
    now = datetime.now()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)

    # Remove old entries
    request_counts[client_ip] = [
        t for t in request_counts[client_ip] if t > window_start
    ]

    # Check limit
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False

    request_counts[client_ip].append(now)
    return True

# --- Request/Response models ---
class ChatRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    max_tokens: int = 500
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_generated: int
    duration_seconds: float
    timestamp: str

# --- Endpoints ---
@app.get("/health")
async def health_check():
    """Check if the server and Ollama are healthy."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "status": "healthy" if ollama_ok else "degraded",
        "ollama_connected": ollama_ok,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def list_models():
    """List available models."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = r.json().get("models", [])
        return {
            "models": [
                {
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / (1024**3), 2)
                }
                for m in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Send a prompt to the model and get a response."""
    client_ip = req.client.host

    # Rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS} seconds."
        )

    # Log the request
    logger.info(f"Request from {client_ip}: model={request.model}, prompt_length={len(request.prompt)}")

    # Call Ollama
    start_time = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "num_predict": request.max_tokens,
                    "temperature": request.temperature
                }
            },
            timeout=120
        )
        r.raise_for_status()
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is it running?")
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Ollama request timed out")

    duration = time.time() - start_time
    result = r.json()

    # Log the response
    tokens = result.get("eval_count", 0)
    logger.info(f"Response: {tokens} tokens in {duration:.2f}s")

    return ChatResponse(
        response=result["response"],
        model=request.model,
        tokens_generated=tokens,
        duration_seconds=round(duration, 3),
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
