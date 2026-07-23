apt update
apt install -y zip vim pv tmux rsync libnuma-dev lsof

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

export UV_CACHE_DIR=/root/.cache/uv
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub 