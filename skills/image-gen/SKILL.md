# Image Generation Service Example

A complete example of deploying a Stable Diffusion image generation service.

```python
import modal
import io
import base64

# --- Configuration ---
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
GPU_TYPE = "A10"

# --- Image Definition ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.1.0",
        "diffusers==0.25.0",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "Pillow",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("image-generation", image=image)

# --- Model Cache ---
model_volume = modal.Volume.from_name("sdxl-cache", create_if_missing=True)
MODEL_PATH = "/models"

# --- Download Model ---
@app.function(
    volumes={MODEL_PATH: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    cpu=4,
    memory=16384,
)
def download_model():
    from diffusers import DiffusionPipeline
    import torch
    import os
    
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN"),
    )
    pipe.save_pretrained(f"{MODEL_PATH}/{MODEL_ID}")
    model_volume.commit()

# --- Generation Service ---
@app.cls(
    gpu=GPU_TYPE,
    volumes={MODEL_PATH: model_volume},
    container_idle_timeout=120,
)
class ImageGenerator:
    @modal.enter()
    def load_pipeline(self):
        from diffusers import DiffusionPipeline
        import torch
        
        self.pipe = DiffusionPipeline.from_pretrained(
            f"{MODEL_PATH}/{MODEL_ID}",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to("cuda")
        
        # Optional: Enable memory optimizations
        self.pipe.enable_model_cpu_offload()
    
    @modal.method()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> bytes:
        import torch
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

# --- Web API ---
@app.function()
@modal.fastapi_endpoint(method="POST", docs=True)
def generate_image(body: dict) -> dict:
    generator = ImageGenerator()
    
    image_bytes = generator.generate.remote(
        prompt=body["prompt"],
        negative_prompt=body.get("negative_prompt", ""),
        width=body.get("width", 1024),
        height=body.get("height", 1024),
        num_inference_steps=body.get("steps", 30),
        guidance_scale=body.get("guidance_scale", 7.5),
        seed=body.get("seed"),
    )
    
    # Return as base64
    image_b64 = base64.b64encode(image_bytes).decode()
    return {"image": image_b64}

# --- Direct Image Response ---
@app.function()
@modal.fastapi_endpoint(method="GET")
def generate_image_direct(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
):
    from fastapi.responses import Response
    
    generator = ImageGenerator()
    image_bytes = generator.generate.remote(
        prompt=prompt,
        width=width,
        height=height,
    )
    
    return Response(content=image_bytes, media_type="image/png")

# --- Batch Generation ---
@app.function(timeout=3600)
def generate_batch(prompts: list[str], output_dir: str = "/output"):
    """Generate multiple images and save to volume."""
    volume = modal.Volume.from_name("generated-images", create_if_missing=True)
    
    generator = ImageGenerator()
    
    for i, prompt in enumerate(prompts):
        image_bytes = generator.generate.remote(prompt=prompt)
        
        with volume.batch_upload() as batch:
            batch.put_file(
                io.BytesIO(image_bytes),
                f"{output_dir}/image_{i:04d}.png"
            )
    
    return f"Generated {len(prompts)} images"

# --- CLI ---
@app.local_entrypoint()
def main(
    prompt: str = "A beautiful sunset over mountains, digital art",
    output: str = "output.png",
):
    print(f"Generating: {prompt}")
    
    generator = ImageGenerator()
    image_bytes = generator.generate.remote(prompt=prompt)
    
    with open(output, "wb") as f:
        f.write(image_bytes)
    
    print(f"Saved to {output}")
```

## Usage

```bash
# Download model first
modal run image_gen.py::download_model

# Generate image via CLI
modal run image_gen.py --prompt "A cat astronaut" --output cat.png

# Deploy API
modal deploy image_gen.py

# Generate via API
curl "https://your-workspace--image-generation-generate-image-direct.modal.run?prompt=A%20cat%20astronaut" \
  --output cat.png
```
