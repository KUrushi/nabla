#!/usr/bin/env python3
"""PyTree チェックポイントの相互運用フィクスチャを作る（issue #206）。

nabla のライブラリ本体は Python に依存しない（CLAUDE.md）。これはフィクスチャを作る
ときにだけ使う開発用スクリプトで、venv 自体はコミットしない。

    python3 -m venv /tmp/venv && /tmp/venv/bin/pip install safetensors==0.4.5 numpy
    /tmp/venv/bin/python tests/fixtures/checkpoint/generate.py

Flax の 2 層 MLP（nn.Dense(4) → relu → nn.Dense(2)）の variables を、
`flax.traverse_util.flatten_dict(variables, sep=".")` と同じ規則（入れ子の dict のキーを
"." でつなぐ）で平坦化し、flax-mlp.safetensors に書く。Flax 本体は入れない（JAX が
重く、規則は dict のキーをつなぐだけなので同じ規則をここで書く）。Dense の kernel は
(入力, 出力) の形で、y = x @ kernel + bias。
同じ重みと入力 X の順伝播の結果 Y を numpy で計算して出力する。値は
tests/nn/checkpoint-test.lisp の定数と一致させる。
"""

import os

import numpy as np
from safetensors.numpy import save

HERE = os.path.dirname(os.path.abspath(__file__))

variables = {
    "params": {
        "Dense_0": {
            "kernel": (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0) / 8.0,
            "bias": np.array([0.5, -0.25, 0.0, 0.125], dtype=np.float32),
        },
        "Dense_1": {
            "kernel": (np.arange(8, dtype=np.float32).reshape(4, 2) - 3.0) / 4.0,
            "bias": np.array([0.25, -0.5], dtype=np.float32),
        },
    }
}
X = np.array([[1.0, -2.0, 0.5], [0.0, 1.0, 3.0]], dtype=np.float32)


def flatten_dict(tree, sep=".", prefix=""):
    """flax.traverse_util.flatten_dict(tree, sep=sep) と同じ結果（キーは文字列）。"""
    out = {}
    for key, value in tree.items():
        path = prefix + sep + key if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, sep, path))
        else:
            out[path] = value
    return out


def forward(p):
    h = np.maximum(X @ p["Dense_0"]["kernel"] + p["Dense_0"]["bias"], 0.0)
    return h @ p["Dense_1"]["kernel"] + p["Dense_1"]["bias"]


flat = flatten_dict(variables)
with open(os.path.join(HERE, "flax-mlp.safetensors"), "wb") as f:
    f.write(save(flat, metadata={"format": "flax"}))
print("names:", sorted(flat))
print("Y =", forward(variables["params"]).tolist())
