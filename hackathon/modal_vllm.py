"""Serve an OpenAI-compatible vLLM endpoint on Modal for the Colosseum runs (new weekend work).

Deploy:  modal secret create colosseum-vllm VLLM_API_KEY=<any-random-string>
         modal deploy hackathon/modal_vllm.py
Prints a URL like https://<workspace>--colosseum-vllm-serve.modal.run ; use <URL>/v1 as base_url.
"""
import subprocess
import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers<5", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("colosseum-vllm")


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("colosseum-vllm")],
    scaledown_window=10 * 60,
    timeout=24 * 60 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve():
    subprocess.Popen(
        "vllm serve " + MODEL + " --host 0.0.0.0 --port 8000 --max-model-len 16000"
        " --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes"
        " --api-key $VLLM_API_KEY",
        shell=True,
    )
