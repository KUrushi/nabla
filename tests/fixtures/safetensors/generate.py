#!/usr/bin/env python3
"""safetensors の相互運用フィクスチャを作る（issue #198）。

nabla のライブラリ本体は Python に依存しない（CLAUDE.md）。これは、フィクスチャを
作るときにだけ使う開発用スクリプトで、venv 自体はコミットしない。

    python3 -m venv /tmp/venv && /tmp/venv/bin/pip install safetensors==0.4.5
    /tmp/venv/bin/python tests/fixtures/safetensors/generate.py

1. python-written.safetensors を書く（Python の safetensors が書いたファイル。
   tests/nn/safetensors-test.lisp の fixture-entries と同じ内容）。
2. nabla-written.safetensors があれば、Python の safetensors で読み、内容が
   PYTHON_TENSORS と一致することを確かめる（nabla が書いたファイルを Python が
   読める）。このファイルは nabla で作り直してコミットする。作り方は
   tests/fixtures/safetensors/write-nabla-fixture.lisp の冒頭を見る。
"""

import os
import struct
import sys

import safetensors

HERE = os.path.dirname(os.path.abspath(__file__))


def pack(fmt, values):
    return b"".join(struct.pack("<" + fmt, v) for v in values)


# (名前, dtype, shape, バイト列)
PYTHON_TENSORS = [
    ("f32", "F32", [2, 2], pack("f", [1.5, -2.0, 0.25, 3.0])),
    ("f64", "F64", [3], pack("d", [0.1, -2.5, 1e100])),
    ("bf16", "BF16", [3], pack("H", [0x3F80, 0xBF80, 0x4000])),
    ("f16", "F16", [2], pack("H", [0x3C00, 0xC000])),
    ("i32", "I32", [3], pack("i", [-2147483648, 0, 2147483647])),
    ("u32", "U32", [2], pack("I", [0, 4294967295])),
    ("u64", "U64", [2], pack("Q", [0, 18446744073709551615])),
    ("bool", "BOOL", [2, 2], bytes([1, 0, 1, 1])),
    ("scalar", "F32", [], pack("f", [7.0])),
    ("empty", "F32", [0, 3], b""),
    ("layer.0/重", "F32", [1], pack("f", [1.0])),
]
# safetensors.serialize（0.4.5）が受け取る dtype 名
SERIALIZE_DTYPE = {"F32": "float32", "F64": "float64", "BF16": "bfloat16", "F16": "float16",
                   "I32": "int32", "U32": "uint32", "U64": "uint64", "BOOL": "bool"}
METADATA = {"format": "pt", "note": "fixture"}


def main():
    tensors = {n: {"dtype": SERIALIZE_DTYPE[d], "shape": s, "data": b} for n, d, s, b in PYTHON_TENSORS}
    data = safetensors.serialize(tensors, METADATA)
    with open(os.path.join(HERE, "python-written.safetensors"), "wb") as f:
        f.write(bytes(data))

    nabla_path = os.path.join(HERE, "nabla-written.safetensors")
    if not os.path.exists(nabla_path):
        print("nabla-written.safetensors が無いので、Python での読み込みの確認は省く", file=sys.stderr)
        return
    with open(nabla_path, "rb") as f:
        loaded = dict(safetensors.deserialize(f.read()))
    assert set(loaded) == {n for n, *_ in PYTHON_TENSORS}, sorted(loaded)
    for name, dtype, shape, raw in PYTHON_TENSORS:
        got = loaded[name]
        assert got["dtype"] == dtype and got["shape"] == shape, (name, got["dtype"], got["shape"])
        assert bytes(got["data"]) == raw, name
    print("Python の safetensors が nabla-written.safetensors を読めた")


if __name__ == "__main__":
    main()
