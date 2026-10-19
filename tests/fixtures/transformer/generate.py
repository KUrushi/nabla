#!/usr/bin/env python3
"""小さな Transformer の学習を Flax + Optax と突き合わせるフィクスチャを作る。

issue #209。nabla のライブラリ本体・テストの実行時は Python / JAX / Flax に依存しない
（CLAUDE.md）。フィクスチャを作るときにだけ使う開発用スクリプトで、生成物
（init.safetensors, after-1.safetensors, after-5.safetensors, trajectory.lisp）をコミットする。
使ったバージョン: jax 0.10.2, flax 0.12.8, optax 0.2.8, safetensors 0.8.0（f32、x64 は無効）。

モデルは examples/transformer.lisp と同じ（語彙 12、長さ 8、幅 16、2 ヘッド、MLP の幅 32、
pre-LN の 2 ブロック、出力は別の dense）。パラメータ名は Flax の flatten_dict のパスで、
nabla のチェックポイントのパス名と同じ。ただし Flax の blocks_0 は blocks.0 と書く
（nabla のリストの添字）。初期値は numpy で作って init.safetensors に書き、nabla はそれを
load-checkpoint で読む（同じ初期値から始めるため。nabla の PRNG は JAX とビットで一致しない）。

学習: optax.chain(clip_by_global_norm(1.0), adam(1e-2))（eps は Optax の既定 1e-8）、損失は
-sum(targets * log_softmax(logits)) / sum(targets)（targets は次のトークンの one-hot に損失をかける位置の
印を掛けたもの）、バッチは 8 本の系列「x1..x4 SEP x1..x4」（SEP = 11）を STEPS 回、ステップごとに別のものを使う。

出力:
  init.safetensors      初期パラメータ
  after-1.safetensors   1 ステップ後のパラメータ
  after-5.safetensors   5 ステップ後のパラメータ
  trajectory.lisp       (:versions .. :batch-size 8 :steps 20 :sequences (i32 の値、(STEPS 8 9) の順)
                          :losses (f32 のビットパターン、各ステップの更新前の損失))

使い方 (jax, flax, optax, safetensors の入った venv で、リポジトリの根から):
    python3 tests/fixtures/transformer/generate.py
"""

import os
import re

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))

VOCAB, COPY, LENGTH = 12, 4, 8
D_MODEL, HEADS, MLP, LAYERS = 16, 2, 32, 2
BATCH, STEPS = 8, 20
LEARNING_RATE, EPS = 1e-2, 1e-8


class Block(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        y = nn.LayerNorm(name="ln1")(x)
        y = nn.MultiHeadDotProductAttention(num_heads=HEADS, qkv_features=D_MODEL, name="attn")(y, y, mask=mask)
        x = x + y
        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(MLP, name="up")(h)
        h = nn.gelu(h)
        h = nn.Dense(D_MODEL, name="down")(h)
        return x + h


class Transformer(nn.Module):
    @nn.compact
    def __call__(self, tokens):
        pos = self.param("pos", nn.initializers.zeros, (LENGTH, D_MODEL))
        x = nn.Embed(VOCAB, D_MODEL, name="embed")(tokens) + pos
        mask = nn.make_causal_mask(tokens)
        for i in range(LAYERS):
            x = Block(name=f"blocks_{i}")(x, mask)
        x = nn.LayerNorm(name="norm")(x)
        return nn.Dense(VOCAB, name="head")(x)


def flatten(tree, prefix=""):
    out = {}
    for key, value in tree.items():
        path = prefix + "." + key if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, path))
        else:
            out[path] = np.asarray(value)
    return out


def nabla_name(name):
    """Flax の blocks_0.ln1.scale → nabla の blocks.0.ln1.scale。"""
    return re.sub(r"^blocks_(\d+)\.", r"blocks.\1.", name)


def unflatten(flat):
    """nabla のパス名の平坦な辞書 → Flax の入れ子の辞書。"""
    out = {}
    for name, value in flat.items():
        parts = re.sub(r"^blocks\.(\d+)\.", r"blocks_\1.", name).split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = jnp.asarray(value)
    return out


def initial_params(model, rng):
    """Flax で形だけ作り、numpy で値を入れる（バイアスも 0 でない）。"""
    shapes = flatten(model.init(jax.random.PRNGKey(0), jnp.zeros((BATCH, LENGTH), jnp.int32))["params"])
    flat = {}
    for name, shape in sorted((n, a.shape) for n, a in shapes.items()):
        leaf = name.split(".")[-1]
        if leaf == "scale":
            value = 1.0 + 0.1 * rng.standard_normal(shape)
        elif leaf == "bias":
            value = 0.05 * rng.standard_normal(shape)
        elif leaf == "embedding":
            value = rng.standard_normal(shape) / np.sqrt(D_MODEL)
        elif leaf == "pos":
            value = 0.05 * rng.standard_normal(shape)
        else:
            value = rng.standard_normal(shape) * (0.5 / np.sqrt(shape[0]))
        flat[nabla_name(name)] = value.astype(np.float32)
    return flat


def make_sequences(rng):
    """(STEPS, BATCH, 2*COPY+1): x1..xN SEP x1..xN。"""
    data = rng.integers(0, VOCAB - 1, size=(STEPS, BATCH, COPY))
    sep = np.full((STEPS, BATCH, 1), VOCAB - 1)
    return np.concatenate([data, sep, data], axis=-1).astype(np.int32)


def to_batch(sequences):
    tokens = sequences[:, : 2 * COPY]
    targets = np.zeros((sequences.shape[0], 2 * COPY, VOCAB), np.float32)
    for p in range(COPY, 2 * COPY):
        targets[np.arange(sequences.shape[0]), p, sequences[:, p + 1]] = 1.0
    return jnp.asarray(tokens), jnp.asarray(targets)


def main():
    model = Transformer()
    rng = np.random.default_rng(209)
    init_flat = initial_params(model, rng)
    sequences = make_sequences(rng)
    params = unflatten(init_flat)

    def loss_fn(p, tokens, targets):
        logp = jax.nn.log_softmax(model.apply({"params": p}, tokens), axis=-1)
        return -jnp.sum(targets * logp) / jnp.sum(targets)

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE, eps=EPS))
    state = tx.init(params)

    @jax.jit
    def step(p, s, tokens, targets):
        loss, grads = jax.value_and_grad(loss_fn)(p, tokens, targets)
        updates, s = tx.update(grads, s, p)
        return loss, optax.apply_updates(p, updates), s

    def save(name, p):
        flat = {nabla_name(k): v for k, v in flatten(p).items()}
        save_file(flat, os.path.join(HERE, name), metadata={"format": "flax"})

    save_file(init_flat, os.path.join(HERE, "init.safetensors"), metadata={"format": "flax"})
    losses = []
    for k in range(STEPS):
        tokens, targets = to_batch(sequences[k])
        loss, params, state = step(params, state, tokens, targets)
        losses.append(float(loss))
        if k == 0:
            save("after-1.safetensors", params)
        if k == 4:
            save("after-5.safetensors", params)
    print("losses:", [round(x, 4) for x in losses])

    bits = np.asarray(losses, np.float32).view(np.uint32)
    with open(os.path.join(HERE, "trajectory.lisp"), "w") as f:
        f.write(";;;; 小さな Transformer の学習のフィクスチャ（issue #209）。\n")
        f.write(";;;; tests/fixtures/transformer/generate.py で生成した。手で編集しない。\n")
        f.write(f'(:versions "jax {jax.__version__} flax {flax.__version__} optax {optax.__version__}"\n')
        f.write(f" :batch-size {BATCH} :steps {STEPS}\n")
        f.write(" :sequences (" + " ".join(str(int(v)) for v in sequences.reshape(-1)) + ")\n")
        f.write(" :losses (" + " ".join(str(int(b)) for b in bits) + "))\n")


if __name__ == "__main__":
    main()
