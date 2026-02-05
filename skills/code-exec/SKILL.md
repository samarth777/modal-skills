# Code Execution Sandbox Example

A complete example of a secure code execution service for LLM-generated code.

```python
import modal
from typing import Optional

app = modal.App("code-executor")

# --- Sandboxed Execution Image ---
sandbox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "sympy",
        "requests",  # for network-enabled sandbox
    )
)

# --- API Image ---
api_image = modal.Image.debian_slim().pip_install("fastapi[standard]")

# --- Secure Code Executor ---
@app.function(
    image=api_image,
    timeout=120,
)
def execute_code(
    code: str,
    timeout: int = 30,
    allow_network: bool = False,
    memory_mb: int = 512,
) -> dict:
    """Execute arbitrary Python code in a secure sandbox."""
    
    # Get or create app reference
    sandbox_app = modal.App.lookup("code-executor", create_if_missing=True)
    
    # Create sandbox with security constraints
    sb = modal.Sandbox.create(
        image=sandbox_image,
        timeout=timeout,
        memory=memory_mb,
        block_network=not allow_network,
        app=sandbox_app,
    )
    
    try:
        # Write code to file
        with sb.open("/tmp/user_code.py", "w") as f:
            f.write(code)
        
        # Execute code
        p = sb.exec("python", "/tmp/user_code.py", timeout=timeout)
        p.wait()
        
        stdout = p.stdout.read()
        stderr = p.stderr.read()
        
        return {
            "success": p.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": p.returncode,
        }
    except TimeoutError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Execution timed out",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }
    finally:
        sb.terminate()

# --- Interactive REPL Session ---
@app.cls(image=api_image, timeout=3600)
class REPLSession:
    """Maintain a persistent Python REPL session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.sb = None
    
    @modal.enter()
    def create_sandbox(self):
        app = modal.App.lookup("code-executor", create_if_missing=True)
        
        # Create persistent sandbox
        self.sb = modal.Sandbox.create(
            image=sandbox_image,
            timeout=3600,  # 1 hour max
            idle_timeout=300,  # 5 min idle timeout
            block_network=True,
            app=app,
        )
        
        # Initialize Python REPL
        self.sb.exec("python", "-c", "import sys; sys.ps1 = '>>> '").wait()
    
    @modal.method()
    def execute(self, code: str) -> dict:
        """Execute code in the persistent session."""
        # Write code to temp file
        with self.sb.open("/tmp/code.py", "w") as f:
            f.write(code)
        
        # Execute
        p = self.sb.exec("python", "/tmp/code.py", timeout=30)
        p.wait()
        
        return {
            "stdout": p.stdout.read(),
            "stderr": p.stderr.read(),
            "success": p.returncode == 0,
        }
    
    @modal.method()
    def install_package(self, package: str) -> dict:
        """Install a pip package in the session."""
        p = self.sb.exec("pip", "install", package, timeout=120)
        p.wait()
        
        return {
            "success": p.returncode == 0,
            "output": p.stdout.read() + p.stderr.read(),
        }
    
    @modal.method()
    def list_files(self, path: str = "/tmp") -> list[str]:
        """List files in the sandbox."""
        return list(self.sb.ls(path))
    
    @modal.method()
    def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        with self.sb.open(path, "r") as f:
            return f.read()
    
    @modal.method()
    def write_file(self, path: str, content: str) -> bool:
        """Write a file to the sandbox."""
        with self.sb.open(path, "w") as f:
            f.write(content)
        return True
    
    @modal.exit()
    def cleanup(self):
        if self.sb:
            self.sb.terminate()

# --- Web API ---
@app.function(image=api_image)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    web_app = FastAPI(title="Code Execution API")
    
    class ExecuteRequest(BaseModel):
        code: str
        timeout: int = 30
        allow_network: bool = False
    
    class ExecuteResponse(BaseModel):
        success: bool
        stdout: str
        stderr: str
        returncode: int
    
    @web_app.post("/execute", response_model=ExecuteResponse)
    async def execute_endpoint(request: ExecuteRequest):
        result = execute_code.remote(
            code=request.code,
            timeout=request.timeout,
            allow_network=request.allow_network,
        )
        return result
    
    class REPLRequest(BaseModel):
        session_id: str
        code: str
    
    @web_app.post("/repl/execute")
    async def repl_execute(request: REPLRequest):
        session = REPLSession(request.session_id)
        result = session.execute.remote(request.code)
        return result
    
    @web_app.post("/repl/install")
    async def repl_install(session_id: str, package: str):
        session = REPLSession(session_id)
        result = session.install_package.remote(package)
        return result
    
    return web_app

# --- Batch Code Execution ---
@app.function(image=api_image, timeout=3600)
def execute_batch(code_snippets: list[dict]) -> list[dict]:
    """Execute multiple code snippets in parallel."""
    
    def run_one(snippet: dict) -> dict:
        return execute_code.remote(
            code=snippet["code"],
            timeout=snippet.get("timeout", 30),
            allow_network=snippet.get("allow_network", False),
        )
    
    # Run in parallel using map
    results = list(execute_code.map(
        [s["code"] for s in code_snippets],
        [s.get("timeout", 30) for s in code_snippets],
        [s.get("allow_network", False) for s in code_snippets],
    ))
    
    return results

# --- LLM Integration Example ---
@app.function(
    image=api_image,
    secrets=[modal.Secret.from_name("openai-secret")],
)
def llm_code_agent(task: str) -> dict:
    """Use an LLM to generate and execute code."""
    import os
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Generate code
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """You are a Python code generator. 
                Generate only Python code that solves the given task.
                The code should print its result to stdout.
                Do not include any explanation, only code."""
            },
            {"role": "user", "content": task}
        ],
    )
    
    generated_code = response.choices[0].message.content
    
    # Clean up code (remove markdown if present)
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0]
    
    # Execute code
    result = execute_code.remote(
        code=generated_code,
        timeout=60,
        allow_network=False,
    )
    
    return {
        "task": task,
        "generated_code": generated_code,
        "execution_result": result,
    }

# --- CLI ---
@app.local_entrypoint()
def main(code: str = "print('Hello from sandbox!')"):
    result = execute_code.remote(code)
    
    print("=== Execution Result ===")
    print(f"Success: {result['success']}")
    print(f"Return code: {result['returncode']}")
    print(f"\n--- stdout ---\n{result['stdout']}")
    if result['stderr']:
        print(f"\n--- stderr ---\n{result['stderr']}")
```

## Usage

```bash
# Simple execution
modal run code_executor.py --code "print(sum(range(100)))"

# Deploy API
modal deploy code_executor.py

# Execute via API
curl -X POST https://your-workspace--code-executor-api.modal.run/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "import numpy as np; print(np.random.rand(5))", "timeout": 30}'

# LLM agent
modal run code_executor.py::llm_code_agent --task "Calculate the first 20 Fibonacci numbers"
```

## Security Considerations

1. **Network isolation**: `block_network=True` prevents outbound connections
2. **Timeout limits**: Prevent infinite loops
3. **Memory limits**: Prevent memory exhaustion
4. **Fresh containers**: Each execution gets a clean environment
5. **No Modal access**: Sandboxes can't access other Modal resources by default
