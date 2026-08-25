#!/bin/bash

# Start timer
start=$(date +%s)

# Make python venv
python3 -m venv --without-pip .venv
source .venv/bin/activate

# Install pip manually
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

python -m pip install torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu

# Install datasets, numpy, and Hugging Face model utilities
pip3 install datasets
pip3 install numpy
pip3 install transformers
pip3 install einops

# Download the tokenizer once into a per-user cache.  The model weights are
# not downloaded here; only the tokenizer files needed by visualize_output.py
# are saved.
tokenizer_repo="${APSS26_TOKENIZER_REPO:-microsoft/Phi-tiny-MoE-instruct}"
user_home_dir="$(getent passwd "$(id -u)" | cut -d: -f6)"
cache_root="${XDG_CACHE_HOME:-${user_home_dir}/.cache}"
tokenizer_dir="${APSS26_TOKENIZER_DIR:-${cache_root}/apss26/phi-tiny-tokenizer}"
export APSS26_TOKENIZER_DIR="$tokenizer_dir"
TOKENIZER_REPO="$tokenizer_repo" TOKENIZER_TARGET="$tokenizer_dir" python3 - <<'PY'
import os
from pathlib import Path
from transformers import AutoTokenizer

repo = os.environ["TOKENIZER_REPO"]
target = Path(os.environ["TOKENIZER_TARGET"])
marker = target / ".ready"
if marker.exists():
    print(f"Tokenizer already available at {target}")
else:
    print(f"Downloading tokenizer {repo} to {target}...")
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    target.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(target)
    marker.write_text(repo + "\n")
    print(f"Tokenizer saved to {target}")
PY

# End timer
end=$(date +%s)

# Print time in minutes and seconds
minutes=$(( (end-start) / 60 ))
seconds=$(( (end-start) % 60 ))
echo "Time taken: $minutes minutes and $seconds seconds"
