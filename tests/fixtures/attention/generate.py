#!/usr/bin/env python3
"""multi-head attention を Flax と突き合わせるフィクスチャを作る。

issue #208。nabla のライブラリ本体・テストの実行時は Python / JAX / Flax に依存しない
（CLAUDE.md）。フィクスチャを作るときにだけ使う開発用スクリプトで、生成した
flax-attention.lisp をコミットする。

各ケースは、固定のパラメータ・入力・マスク・余接線 ct を Flax の
nn.MultiHeadDotProductAttention（deterministic=True）に渡し、
    loss = sum(out * ct)
の出力 out と、パラメータ・入力についての勾配を記録する（dropout は nabla の乱数が JAX と
ビット単位で一致しないのでフィクスチャでは比べない）。f32（jax_enable_x64 は無効）。
パラメータ名は Flax の flatten_dict のパス（query.kernel など）をそのまま使う。

出力形式: 1つの s式。配列は (名前 形状 値) で、float は f32 の u32 ビットパターン、
整数（mask の 0/1）はそのまま。
  (:versions "jax x.y.z flax a.b.c"
   :cases ((:name "self" :num-heads 2 :arrays (("query.kernel" (6 2 3) (...)) ...)) ...)
   :init-stats ((:query-kernel (64 4 16) std maxabs) ...))

使い方 (jax[cpu]==0.10.2 と flax の入った venv で):
    python3 tests/fixtures/attention/generate.py > tests/fixtures/attention/flax-attention.lisp
"""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def bits(x):
    return [int(v) for v in np.asarray(x, dtype=np.float32).view(np.uint32).reshape(-1)]


def arr(name, x):
    x = np.asarray(x)
    shape = " ".join(str(d) for d in x.shape)
    if np.issubdtype(x.dtype, np.integer) or x.dtype == np.bool_:
        values = [int(v) for v in x.reshape(-1)]
    else:
        values = bits(x)
    return f'("{name}" ({shape}) ({" ".join(str(v) for v in values)}))'


def rnd(rng, *shape, scale=1.0):
    return (scale * rng.standard_normal(shape)).astype(np.float32)


def nest(flat):
    out = {}
    for key, value in flat.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def make_params(rng, in_q, in_kv, heads, qkv, out):
    d = qkv // heads
    p = {}
    for name, fan in [("query", in_q), ("key", in_kv), ("value", in_kv)]:
        p[f"{name}.kernel"] = rnd(rng, fan, heads, d, scale=0.5)
        p[f"{name}.bias"] = rnd(rng, heads, d, scale=0.3)
    p["out.kernel"] = rnd(rng, heads, d, out, scale=0.5)
    p["out.bias"] = rnd(rng, out, scale=0.3)
    return p


def case(name, rng, x, kv, heads, qkv, out, mask=None):
    """kv が None なら自己 attention（inputs_q だけ渡す）。"""
    layer = nn.MultiHeadDotProductAttention(num_heads=heads, qkv_features=qkv, out_features=out,
                                            deterministic=True)
    params = make_params(rng, x.shape[-1], (x if kv is None else kv).shape[-1], heads, qkv, out)
    flat = {k: jnp.asarray(v) for k, v in params.items()}
    m = None if mask is None else jnp.asarray(mask)

    def fwd(p, x_, kv_):
        if kv_ is None:
            return layer.apply({"params": nest(p)}, x_, mask=m)
        return layer.apply({"params": nest(p)}, x_, kv_, mask=m)

    o = fwd(flat, jnp.asarray(x), None if kv is None else jnp.asarray(kv))
    ct = rnd(rng, *o.shape)

    def loss(p, x_, kv_):
        return jnp.sum(fwd(p, x_, kv_) * ct)

    if kv is None:
        gp, gx = jax.grad(loss, argnums=(0, 1))(flat, jnp.asarray(x), None)
        gkv = None
    else:
        gp, gx, gkv = jax.grad(loss, argnums=(0, 1, 2))(flat, jnp.asarray(x), jnp.asarray(kv))
    items = [arr(k, v) for k, v in params.items()] + [arr("x", x)]
    if kv is not None:
        items.append(arr("kv", kv))
    if mask is not None:
        items.append(arr("mask", np.asarray(mask).astype(np.int32)))
    items += [arr("ct", ct), arr("out", o)]
    items += [arr("grad." + k, gp[k]) for k in params]
    items.append(arr("grad.x", gx))
    if gkv is not None:
        items.append(arr("grad.kv", gkv))
    return f'(:name "{name}" :num-heads {heads} :arrays ({" ".join(items)}))'


def main():
    rng = np.random.default_rng(208)
    cases = []

    x = rnd(rng, 2, 4, 6)
    cases.append(case("self", rng, x, None, 2, 6, 6))
    cases.append(case("self-causal", rng, x, None, 2, 6, 6, nn.make_causal_mask(x[..., 0])))

    # padding: バッチ 0 は長さ 3、バッチ 1 は全部 padding（すべてのキーが隠れた行は一様な重みになる）
    valid = np.array([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=np.float32)
    cases.append(case("self-padding", rng, x, None, 2, 6, 6,
                      nn.make_attention_mask(np.ones_like(valid), valid)))

    # 交差 attention: 入力の幅が違う、qkv 幅 8 を 4 ヘッド、出力幅 4、長さ 3 と 5
    q_in = rnd(rng, 2, 3, 5)
    kv_in = rnd(rng, 2, 5, 7)
    cases.append(case("cross", rng, q_in, kv_in, 4, 8, 4))
    kv_valid = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=np.float32)
    cases.append(case("cross-padding", rng, q_in, kv_in, 4, 8, 4,
                      nn.make_attention_mask(np.ones((2, 3), np.float32), kv_valid)))

    # バッチ軸なし（rank 2）
    x2 = rnd(rng, 5, 4)
    cases.append(case("self-rank2-causal", rng, x2, None, 2, 4, 4, nn.make_causal_mask(x2[..., 0])))

    # バッチ軸が2つ。因果マスクと padding マスクを combine_masks で合わせる
    x4 = rnd(rng, 2, 3, 4, 6)
    pad = (rng.random((2, 3, 4)) > 0.25).astype(np.float32)
    pad[..., 0] = 1.0
    both = nn.combine_masks(nn.make_causal_mask(x4[..., 0]), nn.make_attention_mask(pad, pad))
    cases.append(case("self-two-batch-dims-causal-padding", rng, x4, None, 3, 6, 6, both))

    key = jax.random.PRNGKey(208)
    layer = nn.MultiHeadDotProductAttention(num_heads=4, qkv_features=64, deterministic=True)
    p = layer.init(key, jnp.zeros((1, 3, 64)))["params"]
    q_kernel, out_kernel = p["query"]["kernel"], p["out"]["kernel"]
    shape = lambda a: "(" + " ".join(str(d) for d in a.shape) + ")"
    stats = [
        f"(:query-kernel {shape(q_kernel)} {float(q_kernel.std())!r} {float(jnp.abs(q_kernel).max())!r})",
        f"(:out-kernel {shape(out_kernel)} {float(out_kernel.std())!r} {float(jnp.abs(out_kernel).max())!r})",
    ]
    print(";;;; Flax (flax.linen) の MultiHeadDotProductAttention との突き合わせ用フィクスチャ（issue #208）。")
    print(";;;; 生成: tests/fixtures/attention/generate.py。形式は generate.py の docstring。")
    print(f'(:versions "jax {jax.__version__} flax {flax.__version__}"')
    print(" :cases (")
    for c in cases:
        print("  " + c)
    print(" )")
    print(" :init-stats (")
    for s in stats:
        print("  " + s)
    print(" ))")


if __name__ == "__main__":
    main()
