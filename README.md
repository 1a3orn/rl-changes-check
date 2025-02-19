# rl-changes-check

To prep:
curl -LsSf https://astral.sh/uv/install.sh | sh
source .local/bin/env
uv venv myenv --python 3.12 --seed
source myenv/bin/activate

uv pip install vllm


# Best
Agentica: no boxed instructions, temperature 0.8
DeepSeek: no boxed instructions, temperature 0.8
Qwen: with boxed instructions, temperature 0.4