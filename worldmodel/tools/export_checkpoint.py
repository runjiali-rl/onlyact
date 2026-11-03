"""Turn a checkpoint written by train.py (trainable_step*.pt) into the released world-model format read by generate.py:
a safetensors file holding only the trainable tensors (LoRA weights and label_emb, float32), with no metadata.
Every tensor is checked for exact equality after writing.

    python tools/export_checkpoint.py logs/justact/checkpoints/trainable_step020000_end.pt checkpoints/worldmodel/justact.safetensors
"""
import argparse
import hashlib

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    sd = torch.load(a.src, map_location="cpu", weights_only=True)["state_dict"]
    save_file({k: v.contiguous() for k, v in sd.items()}, a.dst)
    back = load_file(a.dst)
    assert set(back) == set(sd) and all(back[k].dtype == v.dtype and torch.equal(back[k], v) for k, v in sd.items())
    with open(a.dst, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"{len(sd)} tensors -> {a.dst} (sha256 {digest})")


if __name__ == "__main__":
    main()
