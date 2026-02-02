# Modal Web Endpoints Reference

Detailed reference for creating web APIs and endpoints on Modal.

## Endpoint Types

### FastAPI Endpoint (Recommended)

Simple function-based endpoints:

```python
import modal

app = modal.App("web-api")
image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=image)
@modal.fastapi_endpoint()
def hello(name: str = "World") -> dict:
    return {"message": f"Hello, {name}!"}

@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=True)
def process(body: dict) -> dict:
    return {"received": body}
```

### ASGI App (Full FastAPI/Starlette)

For complex applications with multiple routes:

```python
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    
    web_app = FastAPI()
    
    @web_app.get("/")
    def root():
        return {"status": "ok"}
    
    @web_app.get("/items/{item_id}")
    def get_item(item_id: int):
        return {"item_id": item_id}
    
    @web_app.post("/items")
    def create_item(item: dict):
        return {"created": item}
    
    return web_app
```

### WSGI App (Flask/Django)

```python
@app.function(image=modal.Image.debian_slim().pip_install("flask"))
@modal.wsgi_app()
def flask_app():
    from flask import Flask, jsonify
    
    web_app = Flask(__name__)
    
    @web_app.route("/")
    def index():
        return jsonify({"status": "ok"})
    
    return web_app
```

### Custom Web Server

For non-Python web servers or custom setups:

```python
@app.function()
@modal.web_server(port=8080)
def custom_server():
    import subprocess
    # Start any web server on port 8080
    subprocess.run(["python", "-m", "http.server", "8080"])
```

## Endpoint Configuration

### HTTP Methods

```python
@modal.fastapi_endpoint(method="GET")      # Default
@modal.fastapi_endpoint(method="POST")
@modal.fastapi_endpoint(method="PUT")
@modal.fastapi_endpoint(method="DELETE")
```

### Documentation

```python
# Enable OpenAPI docs at /docs
@modal.fastapi_endpoint(docs=True)
def documented_endpoint():
    ...
```

### Custom Labels

```python
# Custom URL label
@modal.fastapi_endpoint(label="my-api")
def endpoint():
    ...
# URL: https://workspace--my-api.modal.run
```

### Custom Domains

```python
@modal.fastapi_endpoint(custom_domains=["api.example.com", "api.example.net"])
def multi_domain_endpoint():
    ...
```

### Proxy Authentication

```python
# Require Modal proxy auth tokens
@modal.fastapi_endpoint(requires_proxy_auth=True)
def private_endpoint():
    return {"secret": "data"}
```

Call with:
```bash
curl -H "Modal-Key: $TOKEN_ID" \
     -H "Modal-Secret: $TOKEN_SECRET" \
     https://your-endpoint.modal.run
```

## Streaming Responses

### Server-Sent Events (SSE)

```python
from fastapi.responses import StreamingResponse

@app.function(image=image)
@modal.fastapi_endpoint()
async def stream():
    async def generate():
        for i in range(10):
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Streaming with Map

```python
@app.function(gpu="A100")
def process_chunk(chunk: str) -> str:
    return f"processed: {chunk}"

@app.function(image=image)
@modal.fastapi_endpoint()
async def stream_processing(body: dict):
    from fastapi.responses import StreamingResponse
    
    chunks = body["chunks"]
    
    async def generate():
        async for result in process_chunk.map.aio(chunks):
            yield f"data: {result}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Concurrency

### Handle Multiple Requests

```python
@app.function()
@modal.concurrent(max_inputs=20)  # 20 concurrent requests per container
@modal.asgi_app()
def high_concurrency_app():
    ...
```

## URL Structure

Auto-generated URLs follow this pattern:
```
https://<workspace>-<env-suffix>--<app>-<function>.modal.run
```

Example: `https://myworkspace-prod--my-app-hello.modal.run`

### Ephemeral Apps

Apps run with `modal serve` get a `-dev` suffix:
```
https://myworkspace-prod--my-app-hello-dev.modal.run
```

## Request/Response Handling

### Request Bodies

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def create_item(item: Item) -> dict:
    return {"created": item.dict()}
```

### File Uploads

```python
from fastapi import UploadFile, File

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}
```

### Custom Responses

```python
from fastapi.responses import JSONResponse, HTMLResponse

@app.function(image=image)
@modal.fastapi_endpoint()
def custom_response():
    return JSONResponse(
        content={"message": "hello"},
        headers={"X-Custom-Header": "value"}
    )
```

## Long-Running Requests

### Timeouts

Web endpoints have a 150-second HTTP timeout, but auto-redirect for longer tasks.

### Job Queue Pattern

For very long tasks, use spawn + polling:

```python
@app.function()
def long_task(data: dict) -> dict:
    import time
    time.sleep(300)  # 5 minutes
    return {"result": "done"}

@app.function(image=image)
@modal.asgi_app()
def api():
    from fastapi import FastAPI
    web_app = FastAPI()
    
    @web_app.post("/submit")
    async def submit(data: dict):
        call = long_task.spawn(data)
        return {"call_id": call.object_id}
    
    @web_app.get("/result/{call_id}")
    async def result(call_id: str):
        call = modal.FunctionCall.from_id(call_id)
        try:
            return call.get(timeout=0)
        except TimeoutError:
            return {"status": "pending"}, 202
    
    return web_app
```

## Deployment

```bash
# Development with hot-reload
modal serve app.py

# Production deployment
modal deploy app.py
```

## Getting Endpoint URL

```python
@app.function(image=image)
@modal.fastapi_endpoint()
def my_endpoint():
    # Get own URL
    url = my_endpoint.get_web_url()
    return {"url": url}

# From external code
fn = modal.Function.from_name("my-app", "my_endpoint")
url = fn.get_web_url()
```
