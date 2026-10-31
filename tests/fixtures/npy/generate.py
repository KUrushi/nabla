#!/usr/bin/env python3
""".npy の読み込み（issue #282）用のフィクスチャを作る。

nabla のライブラリ本体・テストの実行時は Python に依存しない（CLAUDE.md）。これは
フィクスチャを作るときにだけ使う開発用スクリプトで、venv 自体はコミットしない。

    python3 -m venv /tmp/venv && /tmp/venv/bin/pip install numpy
    /tmp/venv/bin/python tests/fixtures/npy/generate.py

numpy が書いた *.npy と、読めるファイルの期待値 expected.lisp をこのディレクトリに書く。
expected.lisp は 1 つの s 式で、要素は (ファイル名 種類 (形) (値...))。値は row-major で、
種類が :f32 / :f64 なら浮動小数点のビット列（u32 / u64）、それ以外は整数（bool は 0 / 1）。
error-*.npy は読めない（コンディションになる）ファイルで、期待値は書かない。
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def readable_cases():
    rng = np.random.default_rng(282)
    f32 = rng.standard_normal((2, 3, 4)).astype(np.float32)
    f32[0, 0, :3] = [-0.0, np.inf, np.nan]
    return [
        # (ファイル名, 種類, 配列, ヘッダの版。None は numpy の既定の 1.0)
        ("f32.npy", ":f32", f32, None),
        ("f64.npy", ":f64", np.array([0.1, -2.5, 1e300, -0.0, 5e-324]), None),
        ("i32.npy", ":i32", np.array([[-2**31, -1], [0, 2**31 - 1]], dtype="<i4"), None),
        ("i64.npy", ":i64", np.array([-2**63, 0, 2**63 - 1], dtype="<i8"), None),
        ("u32.npy", ":u32", np.array([0, 2**32 - 1], dtype="<u4"), None),
        ("u64.npy", ":u64", np.array([0, 2**64 - 1], dtype="<u8"), None),
        ("u8.npy", ":u8", np.arange(6, dtype=np.uint8).reshape(2, 3) * 50, None),
        ("i8.npy", ":i8", np.array([-128, 0, 127], dtype=np.int8), None),
        ("bool.npy", ":bool", np.array([[True, False], [False, True]]), None),
        ("scalar.npy", ":f32", np.array(7.5, dtype=np.float32), None),
        ("empty.npy", ":f32", np.zeros((0, 3), dtype=np.float32), None),
        ("version-2.npy", ":f32", np.array([1.0, 2.0, 3.0], dtype=np.float32), (2, 0)),
        ("version-3.npy", ":i32", np.array([4, 5], dtype="<i4"), (3, 0)),
    ]


ERROR_CASES = [
    # (ファイル名, 配列)。どれも nabla では読めない
    ("error-fortran.npy", np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))),
    ("error-big-endian.npy", np.array([1.0, 2.0], dtype=">f4")),
    ("error-object.npy", np.array([1, "a"], dtype=object)),
    ("error-complex.npy", np.array([1 + 2j], dtype="<c8")),
    ("error-structured.npy", np.zeros(2, dtype=[("a", "<f4")])),
]


def values(kind, array):
    flat = array.reshape(-1)
    if kind == ":f32":
        flat = flat.view(np.uint32)
    elif kind == ":f64":
        flat = flat.view(np.uint64)
    return " ".join(str(int(v)) for v in flat)


def save(name, array, version):
    with open(os.path.join(HERE, name), "wb") as f:
        np.lib.format.write_array(f, array, version=version, allow_pickle=True)


def main():
    lines = [";;;; .npy の読み込み（issue #282）のフィクスチャの期待値。",
             f";;;; numpy {np.__version__}。tests/fixtures/npy/generate.py で生成した。手で編集しない。",
             "("]
    for name, kind, array, version in readable_cases():
        save(name, array, version)
        shape = " ".join(str(d) for d in array.shape)
        lines.append(f' ("{name}" {kind} ({shape}) ({values(kind, array)}))')
    lines.append(")")
    with open(os.path.join(HERE, "expected.lisp"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for name, array in ERROR_CASES:
        save(name, array, None)


if __name__ == "__main__":
    main()
