# Modal Sandboxes Reference

Detailed reference for running arbitrary code in isolated containers.

## Overview

Sandboxes are containers you can create, interact with, and terminate at runtime. They're ideal for:

- Executing LLM-generated code
- Running untrusted user code
- Interactive development environments
- Dynamic code analysis

## Basic Usage

```python
import modal

# Get or create an App reference
app = modal.App.lookup("sandbox-app", create_if_missing=True)

# Create a sandbox
sb = modal.Sandbox.create(app=app)

# Execute commands
p = sb.exec("python", "-c", "print('Hello!')")
print(p.stdout.read())  # "Hello!\n"

# Clean up
sb.terminate()
```

## Configuration

### Custom Images

```python
image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "matplotlib")
)

with modal.enable_output():  # Show build logs
    sb = modal.Sandbox.create(
        image=image,
        app=app,
    )
```

### Volumes

```python
volume = modal.Volume.from_name("my-volume", create_if_missing=True)

sb = modal.Sandbox.create(
    volumes={"/data": volume},
    app=app,
)

# Files written to /data persist across sandboxes
sb.exec("bash", "-c", "echo 'hello' > /data/output.txt").wait()
```

### Secrets

```python
secret = modal.Secret.from_dict({"API_KEY": "xxx"})

sb = modal.Sandbox.create(
    secrets=[secret],
    app=app,
)

p = sb.exec("bash", "-c", "echo $API_KEY")
print(p.stdout.read())  # "xxx\n"
```

### GPU Access

```python
sb = modal.Sandbox.create(
    gpu="A100",
    app=app,
)

p = sb.exec("nvidia-smi")
print(p.stdout.read())
```

### Timeouts

```python
sb = modal.Sandbox.create(
    timeout=600,        # Max 10 minutes total
    idle_timeout=60,    # Auto-terminate after 60s idle
    app=app,
)
```

### Working Directory

```python
sb = modal.Sandbox.create(
    workdir="/app",
    app=app,
)
```

## Executing Commands

### Basic Execution

```python
# Simple command
p = sb.exec("python", "-c", "print(1+1)")
p.wait()  # Wait for completion
print(p.returncode)  # 0

# Read output
print(p.stdout.read())  # "2\n"
print(p.stderr.read())  # ""
```

### Streaming Output

```python
p = sb.exec("bash", "-c", "for i in {1..5}; do echo $i; sleep 1; done")

# Stream line by line
for line in p.stdout:
    print(line, end="")
```

### With Timeout

```python
p = sb.exec("sleep", "100", timeout=5)
# Raises TimeoutError after 5 seconds
```

### With Entrypoint

Run a single command as the sandbox's main process:

```python
sb = modal.Sandbox.create(
    "python", "-m", "http.server", "8080",
    app=app,
    timeout=60,
)

# Sandbox runs until command completes or timeout
for line in sb.stdout:
    print(line, end="")
```

## File System Access

### Filesystem API (Alpha)

```python
# Write file
with sb.open("output.txt", "w") as f:
    f.write("Hello, World!")

# Read file
with sb.open("output.txt", "r") as f:
    content = f.read()

# Binary mode
with sb.open("data.bin", "wb") as f:
    f.write(b"\x00\x01\x02")

# Directory operations
sb.mkdir("/app/data")
files = sb.ls("/app")
sb.rm("/app/data/temp.txt")
```

### Using Volumes

```python
# Ephemeral volume (auto-cleanup)
with modal.Volume.ephemeral() as vol:
    # Upload files
    with vol.batch_upload() as batch:
        batch.put_file("local.txt", "/remote.txt")
        batch.put_directory("./local_dir", "/remote_dir")
    
    sb = modal.Sandbox.create(
        volumes={"/data": vol},
        app=app,
    )
    
    # Work with files
    sb.exec("cat", "/data/remote.txt").wait()
    
    sb.terminate()
    sb.wait(raise_on_termination=False)
    
    # Read files after sandbox terminates
    for chunk in vol.read_file("output.txt"):
        print(chunk)
```

### Syncing Volume Changes

For Volumes v2, explicitly sync changes:

```python
sb.exec("bash", "-c", "echo 'data' > /data/file.txt").wait()
sb.exec("sync", "/data").wait()  # Persist changes immediately
```

## Networking

### Port Forwarding with Tunnels

```python
sb = modal.Sandbox.create(
    "python", "-m", "http.server", "8080",
    app=app,
)

tunnel = sb.tunnels()[8080]
print(f"Access at: {tunnel.url}")
```

### Network Isolation

```python
# Block all network access
sb = modal.Sandbox.create(
    block_network=True,
    app=app,
)
```

## Snapshots

### Filesystem Snapshots

Save sandbox state as a new image:

```python
sb = modal.Sandbox.create(app=app)

# Set up environment
sb.exec("pip", "install", "pandas").wait()
sb.exec("mkdir", "/app/data").wait()

# Create snapshot
image = sb.snapshot_filesystem()
sb.terminate()

# Use snapshot for new sandbox
sb2 = modal.Sandbox.create(image=image, app=app)
sb2.exec("python", "-c", "import pandas; print(pandas.__version__)").wait()
```

## Security Best Practices

### For Untrusted Code

```python
sb = modal.Sandbox.create(
    # Restrict Modal API access
    restrict_modal_access=True,
    
    # Block network
    block_network=True,
    
    # Set timeout
    timeout=30,
    
    # Minimal image
    image=modal.Image.debian_slim(),
    
    app=app,
)
```

### LLM Code Execution

```python
def execute_llm_code(code: str) -> dict:
    app = modal.App.lookup("code-executor", create_if_missing=True)
    
    sb = modal.Sandbox.create(
        image=modal.Image.debian_slim().pip_install("numpy", "pandas"),
        timeout=30,
        block_network=True,
        app=app,
    )
    
    try:
        # Write code to file
        with sb.open("/tmp/code.py", "w") as f:
            f.write(code)
        
        # Execute
        p = sb.exec("python", "/tmp/code.py", timeout=10)
        p.wait()
        
        return {
            "stdout": p.stdout.read(),
            "stderr": p.stderr.read(),
            "returncode": p.returncode,
        }
    finally:
        sb.terminate()
```

## Lifecycle Management

### Graceful Shutdown

```python
sb.terminate()  # Request termination
sb.wait(raise_on_termination=False)  # Wait for cleanup
```

### Check Status

```python
if sb.poll() is None:
    print("Still running")
else:
    print(f"Terminated with code: {sb.poll()}")
```

## Async Usage

```python
import asyncio

async def run_async():
    app = await modal.App.lookup.aio("sandbox-app", create_if_missing=True)
    sb = await modal.Sandbox.create.aio(app=app)
    
    p = await sb.exec.aio("python", "-c", "print('async!')")
    output = await p.stdout.read.aio()
    print(output)
    
    await sb.terminate.aio()
```
