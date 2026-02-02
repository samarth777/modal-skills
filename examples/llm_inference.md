# LLM Inference Service Example

A complete example of deploying an LLM inference service on Modal using vLLM.

```python
import modal

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
GPU_TYPE = "A100"

# --- Image Definition ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.6.0",
        "torch==2.4.0",
        "transformers",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
)

app = modal.App("llm-inference", image=image)

# --- Model Cache Volume ---
model_volume = modal.Volume.from_name("llm-model-cache", create_if_missing=True)
MODEL_PATH = "/models"

# --- Download Model (Build Step) ---
@app.function(
    volumes={MODEL_PATH: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def download_model():
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        MODEL_NAME,
        local_dir=f"{MODEL_PATH}/{MODEL_NAME}",
        token=os.environ["HF_TOKEN"],
    )
    model_volume.commit()

# --- Inference Service ---
@app.cls(
    gpu=GPU_TYPE,
    volumes={MODEL_PATH: model_volume},
    container_idle_timeout=300,  # Keep warm for 5 minutes
    allow_concurrent_inputs=10,
)
class LLMService:
    @modal.enter()
    def load_model(self):
        from vllm import LLM
        
        self.llm = LLM(
            model=f"{MODEL_PATH}/{MODEL_NAME}",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
    
    @modal.method()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        from vllm import SamplingParams
        
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text
    
    @modal.method()
    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[str]:
        from vllm import SamplingParams
        
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

# --- Web API ---
@app.function()
@modal.fastapi_endpoint(method="POST", docs=True)
def generate(body: dict) -> dict:
    service = LLMService()
    
    result = service.generate.remote(
        prompt=body["prompt"],
        max_tokens=body.get("max_tokens", 256),
        temperature=body.get("temperature", 0.7),
    )
    
    return {"response": result}

# --- Streaming Endpoint ---
@app.function()
@modal.fastapi_endpoint(method="POST")
async def generate_stream(body: dict):
    from fastapi.responses import StreamingResponse
    
    # For streaming, you'd use vLLM's async engine
    # This is a simplified example
    service = LLMService()
    result = service.generate.remote(
        prompt=body["prompt"],
        max_tokens=body.get("max_tokens", 256),
    )
    
    async def stream():
        # In production, use vLLM's streaming
        for token in result.split():
            yield f"data: {token}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")

# --- CLI ---
@app.local_entrypoint()
def main(prompt: str = "Explain quantum computing in simple terms."):
    print(f"Prompt: {prompt}\n")
    
    service = LLMService()
    response = service.generate.remote(prompt)
    
    print(f"Response:\n{response}")
```

## Usage

```bash
# Download model first
modal run llm_service.py::download_model

# Test locally
modal run llm_service.py --prompt "What is the meaning of life?"

# Deploy
modal deploy llm_service.py

# Call API
curl -X POST https://your-workspace--llm-inference-generate.modal.run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'
```
