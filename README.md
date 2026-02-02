> **Note:** This repository follows the [Agent Skills](https://agentskills.io) standard. For the official Anthropic skills collection, see [anthropics/skills](https://github.com/anthropics/skills).

# Modal Skills

Skills are folders of instructions, scripts, and resources that Claude loads dynamically to improve performance on specialized tasks. This repository contains a comprehensive skill for building serverless Python applications on [Modal](https://modal.com) - a cloud platform for running AI/ML workloads, GPU-accelerated code, web endpoints, scheduled jobs, and batch processing.

For more information about skills, check out:
- [What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills)
- [Using skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [How to create custom skills](https://support.claude.com/en/articles/12512198-creating-custom-skills)

# About This Repository

This skill teaches Claude how to effectively build and deploy applications on Modal's serverless platform. It covers GPU-accelerated inference, web APIs, scheduled jobs, parallel processing, sandboxed code execution, and more.

The skill is self-contained with a `SKILL.md` file containing the main instructions, plus reference documentation and examples for more complex use cases.

## Topics Covered

- **Getting Started**: Installation, authentication, basic app structure
- **Core Concepts**: Apps, Functions, Images, Volumes, Secrets
- **GPU Acceleration**: All GPU types (T4 through B200), multi-GPU, selection guidelines
- **Web Endpoints**: FastAPI, ASGI/WSGI, streaming, custom domains
- **Persistent Storage**: Volumes, cloud bucket mounts
- **Scheduling**: Cron jobs and periodic tasks
- **Parallel Processing**: `.map()`, `.starmap()`, `.spawn()`
- **Sandboxes**: Secure dynamic code execution
- **Classes with Lifecycle**: Stateful services with `@modal.enter()` and `@modal.exit()`
- **Resource Management**: CPU, memory, timeouts, retries
- **Deployment**: CI/CD integration, best practices

# Repository Structure

```
modal-skills/
├── .claude-plugin/
│   └── marketplace.json     # Plugin registration
├── SKILL.md                 # Main skill instructions
├── LICENSE.txt              # MIT License
├── reference/               # Detailed reference documentation
│   ├── gpus.md              # GPU types and configuration
│   ├── images.md            # Container image building
│   ├── sandboxes.md         # Dynamic code execution
│   └── web_endpoints.md     # Web API creation
└── examples/                # Complete working examples
    ├── llm_inference.md     # LLM service with vLLM
    ├── image_generation.md  # Stable Diffusion service
    ├── data_pipeline.md     # ETL pipeline example
    └── code_executor.md     # Secure code sandbox
```

# Try in Claude Code, OpenCode, Copilot CLI

## Claude Code

You can register this repository as a Claude Code Plugin marketplace by running:

```
/plugin marketplace add samarth777/modal-skills
```

Then install the skill:
1. Select `Browse and install plugins`
2. Select `modal-skills`
3. Select `modal`
4. Select `Install now`

Or directly install via:
```
/plugin install modal@modal-skills
```

After installing, you can use the skill by mentioning Modal. For example: "Use the Modal skill to help me deploy a GPU-accelerated LLM inference API"


# Quick Start with Modal

```bash
# Install Modal
pip install modal

# Authenticate
modal setup

# Create your first app
cat > hello.py << 'EOF'
import modal

app = modal.App("hello-modal")

@app.function()
def hello(name: str) -> str:
    return f"Hello, {name}!"

@app.local_entrypoint()
def main():
    print(hello.remote("World"))


# Run it
modal run hello.py
```

# Links

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://github.com/modal-labs/modal-examples)
- [Modal Pricing](https://modal.com/pricing)
- [Agent Skills Specification](https://agentskills.io)
- [Anthropic Skills Repository](https://github.com/anthropics/skills)

# License

MIT License - see [LICENSE.txt](LICENSE.txt)

