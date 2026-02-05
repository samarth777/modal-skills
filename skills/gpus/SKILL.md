# Modal GPU Reference

Detailed reference for GPU acceleration on Modal.

## Available GPUs

| GPU | VRAM | Max Count | Best For |
|-----|------|-----------|----------|
| `T4` | 16 GB | 8 | Budget inference, light training |
| `L4` | 24 GB | 8 | Inference |
| `A10` | 24 GB | 4 | Inference, light training |
| `A100-40GB` | 40 GB | 8 | Training, large model inference |
| `A100-80GB` | 80 GB | 8 | Large models, distributed training |
| `L40S` | 48 GB | 8 | Best cost/performance ratio |
| `H100` | 80 GB | 8 | High-performance training |
| `H200` | 141 GB | 8 | Very large models |
| `B200` | 192 GB | 8 | Largest models, cutting-edge |

## GPU Selection

### Single GPU

```python
@app.function(gpu="A100")
def train():
    import torch
    assert torch.cuda.is_available()
```

### Multiple GPUs

```python
# 8 H100s for large model training
@app.function(gpu="H100:8")
def train_large_model():
    import torch
    device_count = torch.cuda.device_count()  # 8
```

### GPU Fallbacks

Request multiple GPU types in priority order:

```python
@app.function(gpu=["H100", "A100-80GB", "A100-40GB:2"])
def flexible_function():
    # Will try H100 first, then A100-80GB, then 2x A100-40GB
    ...
```

### Specific GPU Variants

```python
# Specific A100 variant
@app.function(gpu="A100-40GB")
def smaller_a100():
    ...

@app.function(gpu="A100-80GB")
def larger_a100():
    ...

# Prevent H100 → H200 auto-upgrade
@app.function(gpu="H100!")
def strict_h100():
    ...
```

## GPU Selection Guidelines

### For Inference

| Model Size | Recommended GPU |
|------------|-----------------|
| < 7B params | `L4`, `T4` |
| 7B-13B params | `L40S`, `A100-40GB` |
| 13B-70B params | `A100-80GB`, `H100` |
| > 70B params | `H100:2+`, `H200`, `B200` |

### For Training

| Training Type | Recommended GPU |
|---------------|-----------------|
| Fine-tuning (LoRA) | `A100-40GB`, `L40S` |
| Full fine-tuning | `A100-80GB`, `H100` |
| Pre-training | `H100:8`, `H200:8` |

## Multi-GPU Training

### PyTorch DDP

```python
@app.function(gpu="A100:4")
def train_ddp():
    import subprocess
    import sys
    subprocess.run(
        ["torchrun", "--nproc_per_node=4", "train.py"],
        check=True
    )
```

### Multi-Node Clusters (Beta)

```python
import modal.experimental

@app.function(gpu="H100:8", timeout=86400)
@modal.experimental.clustered(size=4, rdma=True)  # 4 nodes × 8 GPUs = 32 GPUs
def train_distributed():
    cluster_info = modal.experimental.get_cluster_info()
    
    from torch.distributed.run import run, parse_args
    run(parse_args([
        f"--nnodes={4}",
        f"--node-rank={cluster_info.rank}",
        f"--master-addr={cluster_info.container_ips[0]}",
        "--nproc-per-node=8",
        "--master-port=1234",
        "train.py",
    ]))
```

## CUDA Setup

### Using Pre-installed Drivers

Modal provides CUDA drivers by default. Many libraries work with just pip:

```python
image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="A100", image=image)
def cuda_function():
    import torch
    print(torch.cuda.is_available())  # True
```

### Full CUDA Toolkit

For libraries requiring nvcc or CUDA headers:

```python
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12"
    )
    .entrypoint([])
    .pip_install("flash-attn", "triton")
)

@app.function(gpu="H100", image=image)
def advanced_cuda():
    ...
```

## GPU Metrics

Access GPU metrics from within your function:

```python
@app.function(gpu="A100")
def check_gpu():
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv"],
        capture_output=True, text=True
    )
    print(result.stdout)
```

## Cost Optimization

1. **Right-size your GPU**: Start with smaller GPUs and scale up
2. **Use container idle timeout**: Keep containers warm for repeated requests
3. **Batch requests**: Use `@modal.batched` for throughput
4. **Consider GPU fallbacks**: Accept any available GPU type

```python
@app.cls(
    gpu="L40S",
    container_idle_timeout=300,  # Keep warm for 5 min
)
class InferenceService:
    @modal.enter()
    def load_model(self):
        self.model = load_model()
    
    @modal.batched(max_batch_size=32, wait_ms=100)
    async def predict(self, inputs: list[str]) -> list[str]:
        return self.model.batch_predict(inputs)
```
