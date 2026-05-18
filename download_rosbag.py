# pip install -U huggingface_hub
# you can also download from https://huggingface.co/datasets/Lingpenghaha/Sonarsweep_dataset/tree/main

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lingpenghaha/Sonarsweep_dataset",
    repo_type="dataset",
    allow_patterns="test.bag",
    local_dir="./raw_dataset/test"
)