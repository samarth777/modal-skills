# Modal Images Reference

This document provides detailed reference for building container images in Modal.

## Base Images

### `debian_slim`
Minimal Debian-based image, recommended for most use cases.

```python
image = modal.Image.debian_slim(python_version="3.12")
```

### `micromamba`
For conda/mamba package management.

```python
image = modal.Image.micromamba().micromamba_install(
    "pytorch", "cudatoolkit=11.8",
    channels=["pytorch", "conda-forge"]
)
```

### `from_registry`
Use any public Docker image.

```python
image = modal.Image.from_registry("pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime")

# Add Python to images without it
image = modal.Image.from_registry("ubuntu:22.04", add_python="3.12")
```

### `from_dockerfile`
Build from existing Dockerfile.

```python
image = modal.Image.from_dockerfile("./Dockerfile")
```

### `from_aws_ecr`
Pull from private AWS ECR.

```python
aws_secret = modal.Secret.from_name("my-aws-secret")
image = modal.Image.from_aws_ecr(
    "123456789.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest",
    secret=aws_secret,
)
```

## Image Methods

### Package Installation

```python
image = (
    modal.Image.debian_slim()
    # Python packages
    .pip_install("torch", "transformers")           # Standard pip
    .uv_pip_install("numpy", "pandas")              # Fast uv installer
    .pip_install("flash-attn", gpu="H100")          # With GPU access
    
    # System packages
    .apt_install("git", "ffmpeg", "curl")
    
    # Conda packages
    # (only with micromamba base)
    .micromamba_install("pytorch", channels=["pytorch"])
)
```

### Adding Local Files

```python
image = (
    modal.Image.debian_slim()
    # Add local directory
    .add_local_dir("./config", remote_path="/app/config")
    
    # Add single file
    .add_local_file("./model.py", remote_path="/app/model.py")
    
    # Add Python module (for imports)
    .add_local_python_source("my_module")
)
```

### Environment Configuration

```python
image = (
    modal.Image.debian_slim()
    # Environment variables
    .env({"MY_VAR": "value", "DEBUG": "true"})
    
    # Working directory
    .workdir("/app")
    
    # Custom entrypoint
    .entrypoint(["/usr/bin/my_entrypoint.sh"])
)
```

### Running Commands

```python
image = (
    modal.Image.debian_slim()
    # Shell commands
    .run_commands(
        "git clone https://github.com/user/repo",
        "cd repo && pip install -e ."
    )
    
    # Python function during build
    .run_function(download_models, secrets=[hf_secret])
)
```

## Build Optimization

### Layer Ordering
Order layers from least to most frequently changed:

```python
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")              # 1. System packages (rarely change)
    .pip_install("torch==2.1.0")        # 2. Large dependencies (pinned)
    .pip_install("transformers")        # 3. Application dependencies
    .add_local_python_source("app")     # 4. Application code (changes often)
)
```

### Force Rebuild

```python
# Force specific layer to rebuild
image = (
    modal.Image.debian_slim()
    .pip_install("my-package", force_build=True)
)
```

```bash
# Force all images to rebuild
MODAL_FORCE_BUILD=1 modal run app.py

# Ignore cache without breaking it
MODAL_IGNORE_CACHE=1 modal run app.py
```

### GPU During Build

```python
# Some packages need GPU access during installation
image = (
    modal.Image.debian_slim()
    .pip_install("bitsandbytes", gpu="H100")
)
```

## CUDA Images

For libraries requiring full CUDA toolkit:

```python
cuda_version = "12.8.1"
os_version = "ubuntu24.04"

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{cuda_version}-devel-{os_version}",
        add_python="3.12"
    )
    .entrypoint([])  # Remove base image entrypoint
    .pip_install("torch", "flash-attn")
)
```

## Image Imports

Handle packages that only exist in the container:

```python
image = modal.Image.debian_slim().pip_install("torch", "transformers")

# Imports only happen when container runs
with image.imports():
    import torch
    from transformers import pipeline

@app.function(image=image)
def my_function():
    # torch and pipeline are available here
    model = pipeline("text-generation")
    ...
```
