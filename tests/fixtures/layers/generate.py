#!/usr/bin/env python3
"""dense / layer-norm / embedding / 活性化関数 / sequential を Flax と突き合わせるフィクスチャを作る。

issue #204。nabla のライブラリ本体・テストの実行時は Python / JAX / Flax に依存しない
（CLAUDE.md）。フィクスチャを作るときにだけ使う開発用スクリプトで、生成した
flax-layers.lisp をコミットする。

各ケースは、固定のパラメータ・入力・余接線 ct を Flax の層に渡し、
    loss = sum(out * ct)
の出力 out と、パラメータ・入力についての勾配を記録する（nabla の乱数は JAX とビット単位で
一致しないので、init の値でなく「同じパラメータを与えたときの順伝播と勾配」を比べる）。
init の分布は、Flax の init を1回サンプルした標準偏差と最大絶対値だけを記録する
（:init-stats）。f32（jax_enable_x64 は無効）。

出力形式: 1つの s式。配列は (名前 形状 値) で、float は f32 の u32 ビットパターン、
整数（ids）はそのまま。
  (:versions "jax x.y.z flax a.b.c"
   :cases ((:name "dense" :arrays (("kernel" (5 4) (...)) ...)) ...)
   :init-stats ((:dense-kernel (512 256) std maxabs) ...))

注: JAX / Flax は Linux の Rosetta VM では import できないので、macOS arm64 の venv で生成する（issue #241）。

使い方 (jax[cpu]==0.10.2 と flax の入った venv で):
    python3 tests/fixtures/layers/generate.py > tests/fixtures/layers/flax-layers.lisp
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
    if np.issubdtype(x.dtype, np.integer):
        values = [int(v) for v in x.reshape(-1)]
    else:
        values = bits(x)
    return f'("{name}" ({shape}) ({" ".join(str(v) for v in values)}))'


def rnd(rng, *shape, scale=1.0, shift=0.0):
    return (scale * rng.standard_normal(shape) + shift).astype(np.float32)


def nest(flat):
    out = {}
    for key, value in flat.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def case(name, module, params, x, ct, x_is_float=True):
    """params: {名前: 配列}（入れ子のパスは '.' でつなぐ）。"""

    def loss(p, x_):
        return jnp.sum(module.apply({"params": nest(p)}, x_) * ct)

    flat = {k: jnp.asarray(v) for k, v in params.items()}
    xj = jnp.asarray(x)
    out = module.apply({"params": nest(flat)}, xj)
    if x_is_float:
        gp, gx = jax.grad(loss, argnums=(0, 1))(flat, xj)
    else:
        gp = jax.grad(loss)(flat, xj)
        gx = None
    items = [arr(k, v) for k, v in params.items()] + [arr("x", x), arr("ct", ct), arr("out", out)]
    items += [arr("grad." + k, gp[k]) for k in params]
    if gx is not None:
        items.append(arr("grad.x", gx))
    return f'(:name "{name}" :arrays ({" ".join(items)}))'


class Mlp(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(6, name="0")(x)
        x = nn.relu(x)
        return nn.Dense(3, name="2")(x)


class Act(nn.Module):
    fn: str
    axis: int = -1

    @nn.compact
    def __call__(self, x):
        if self.fn == "relu":
            return nn.relu(x)
        if self.fn == "gelu":
            return nn.gelu(x)
        if self.fn == "softmax":
            return nn.softmax(x, axis=self.axis)
        return nn.log_softmax(x, axis=self.axis)


def extra_cases(rng):
    """issue #241（Flax との差）で足したケース。"""
    cases = []
    cases.append(case("dense-no-bias", nn.Dense(4, use_bias=False), {"kernel": rnd(rng, 5, 4)},
                      rnd(rng, 2, 3, 5), rnd(rng, 2, 3, 4)))

    # layer-norm の use_scale / use_bias / reduction_axes / feature_axes。平均がずれた入力
    # （generate の layer-norm と同じ。許容誤差は layers-test.lisp の %case-tolerance）
    x = rnd(rng, 2, 3, 6, scale=2.0, shift=3.0)
    ct = rnd(rng, 2, 3, 6)
    scale6 = rnd(rng, 6, scale=0.5, shift=1.0)
    bias6 = rnd(rng, 6)
    cases.append(case("layer-norm-no-scale", nn.LayerNorm(epsilon=1e-6, use_scale=False), {"bias": bias6}, x, ct))
    cases.append(case("layer-norm-no-bias", nn.LayerNorm(epsilon=1e-6, use_bias=False), {"scale": scale6}, x, ct))
    cases.append(case("layer-norm-no-scale-no-bias", nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False),
                      {}, x, ct))
    # 群ごとに正規化する軸を変える。reduction_axes=(1, 2)、feature_axes は既定の最後の軸（scale と bias は (5)）
    cases.append(case("layer-norm-reduction-axes", nn.LayerNorm(epsilon=1e-6, reduction_axes=(1, 2)),
                      {"scale": rnd(rng, 5, scale=0.5, shift=1.0), "bias": rnd(rng, 5)},
                      rnd(rng, 2, 3, 4, 5, scale=2.0, shift=3.0), rnd(rng, 2, 3, 4, 5)))
    # 真ん中の軸（-2）で正規化する。scale と bias は最後の軸の大きさ
    cases.append(case("layer-norm-reduction-middle-axis", nn.LayerNorm(epsilon=1e-6, reduction_axes=-2),
                      {"scale": rnd(rng, 5, scale=0.5, shift=1.0), "bias": rnd(rng, 5)},
                      rnd(rng, 4, 3, 5, scale=2.0, shift=3.0), rnd(rng, 4, 3, 5)))
    # feature_axes=(1, 2): scale と bias は入力の軸 1, 2 の大きさ (3, 4)
    cases.append(case("layer-norm-feature-axes",
                      nn.LayerNorm(epsilon=1e-6, reduction_axes=(1, 2), feature_axes=(1, 2)),
                      {"scale": rnd(rng, 3, 4, scale=0.5, shift=1.0), "bias": rnd(rng, 3, 4)},
                      rnd(rng, 2, 3, 4, scale=2.0, shift=3.0), rnd(rng, 2, 3, 4)))
    return cases


def main():
    rng = np.random.default_rng(204)
    cases = []

    cases.append(case("dense", nn.Dense(4), {"kernel": rnd(rng, 5, 4), "bias": rnd(rng, 4)},
                      rnd(rng, 2, 3, 5), rnd(rng, 2, 3, 4)))
    cases.append(case("dense-rank1", nn.Dense(4), {"kernel": rnd(rng, 5, 4), "bias": rnd(rng, 4)},
                      rnd(rng, 5), rnd(rng, 4)))

    # 平均がずれた入力（Flax の fast variance と同じ式で比べる）
    cases.append(case("layer-norm", nn.LayerNorm(epsilon=1e-6),
                      {"scale": rnd(rng, 6, scale=0.5, shift=1.0), "bias": rnd(rng, 6)},
                      rnd(rng, 2, 3, 6, scale=2.0, shift=3.0), rnd(rng, 2, 3, 6)))
    # 分散が epsilon と同程度（epsilon の効き。平均は 0 にして fast variance の桁落ちを避ける）
    x = rnd(rng, 2, 4, scale=1e-3)
    cases.append(case("layer-norm-tiny-variance", nn.LayerNorm(epsilon=1e-6),
                      {"scale": rnd(rng, 4, scale=0.5, shift=1.0), "bias": rnd(rng, 4)}, x,
                      rnd(rng, 2, 4)))

    ids = rng.integers(0, 7, size=(2, 5)).astype(np.int32)
    ids[0, 1] = ids[0, 0]  # 同じ id の繰り返し（勾配が足し合わされる）
    cases.append(case("embedding", nn.Embed(7, 4), {"embedding": rnd(rng, 7, 4)}, ids,
                      rnd(rng, 2, 5, 4), x_is_float=False))

    x = rnd(rng, 3, 5, scale=2.0)
    for fn, axis in [("relu", -1), ("gelu", -1), ("softmax", -1), ("softmax", 0), ("log-softmax", -1)]:
        name = fn if axis == -1 else f"{fn}-axis{axis}"
        cases.append(case(name, Act(fn.replace("-", "_"), axis), {}, x, rnd(rng, 3, 5)))

    cases.append(case("mlp", Mlp(),
                      {"0.kernel": rnd(rng, 5, 6), "0.bias": rnd(rng, 6),
                       "2.kernel": rnd(rng, 6, 3), "2.bias": rnd(rng, 3)},
                      rnd(rng, 4, 5), rnd(rng, 4, 3)))

    # issue #241: 既存のケースの乱数列を変えないよう、別のシードで足す
    rng = np.random.default_rng(241)
    cases += extra_cases(rng)

    key = jax.random.PRNGKey(204)
    kernel = nn.Dense(256).init(key, jnp.zeros((1, 512)))["params"]["kernel"]
    embedding = nn.Embed(1000, 64).init(key, jnp.zeros((1,), jnp.int32))["params"]["embedding"]
    stats = [
        f"(:dense-kernel (512 256) {float(kernel.std())!r} {float(jnp.abs(kernel).max())!r})",
        f"(:embedding (1000 64) {float(embedding.std())!r} {float(jnp.abs(embedding).max())!r})",
    ]
    print(";;;; Flax (flax.linen) との突き合わせ用フィクスチャ（issue #204）。")
    print(";;;; 生成: tests/fixtures/layers/generate.py。形式は generate.py の docstring。")
    print(";;;; issue #241 のケースを足した版は macOS arm64（Apple Silicon）で JAX 0.10.2 / Flax 0.12.8 / x64 無効で生成した。")
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
