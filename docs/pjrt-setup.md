# PJRT プラグインの取得（issue #78）

nabla は PJRT（XLA の外部向け C API）のプラグインを、IREE と同じく固定した版で
再現可能に手に入れる。Python は要らない（curl + unzip + sha256sum だけ）。

## 固定している版

`third_party/pjrt.lock` が正本。x86_64 Linux の manylinux wheel（zip）。

| 種類 | wheel | 中の .so | PJRT API |
|---|---|---|---|
| CPU | `xla-cpu-pjrt` 0.0.1（OpenXLA。この版しか無い） | `xla_plugins/xla_cpu_pjrt/xla_cpu_pjrt.so`（約 260 MB） | 0.81 |
| CUDA | `jax-cuda13-pjrt` 0.11.2 | `jax_plugins/xla_cuda13/xla_cuda_plugin.so`（約 410 MB） | 0.115 |

jaxlib には Lisp から使える CPU 用の PJRT プラグインが入っていない（CPU
クライアントは `GetPjrtApi` を export していない）ため、`xla-cpu-pjrt` を使う。
CUDA の `libcuda.so.1` は遅延ロードなので、GPU が無くてもロードはできる。

ヘッダ `third_party/pjrt/pjrt_c_api.h` は PJRT API 0.116（出所の commit・
ライセンスは `third_party/pjrt/README.md`）。プラグインは自分の版以上の
`struct_size` を受け付けるので、新しいヘッダは 0.81 の CPU プラグインでも使える
（逆は不可）。

## ディスク使用量

展開後の `.so` だけで CPU 約 260 MB、CUDA も入れると約 670 MB（wheel は展開後に消す。ダウンロード中は一時的に wheel の分（60〜130 MB）が加わる）。CI は `NABLA_PJRT_HOME` をキャッシュする。

## 取得

```sh
scripts/fetch-pjrt.sh            # CPU プラグインだけ
scripts/fetch-pjrt.sh --cuda     # CUDA プラグインも
scripts/fetch-pjrt.sh --keep-wheels
```

- インストール先は `NABLA_PJRT_HOME`（既定 `~/.local/share/nabla/pjrt-0.0.1`）。
  `cpu/xla_cpu_pjrt.so` と `cuda/xla_cuda_plugin.so` が置かれる
- wheel は `NABLA_PJRT_WHEEL_DIR`（既定 `${XDG_CACHE_HOME:-~/.cache}/nabla/pjrt-wheel`）
  にダウンロードして sha256 を `third_party/pjrt.lock` と照合し（不一致は失敗）、
  `.so` だけを取り出して wheel を消す。TLS 検証は無効化しない
- 冪等。展開済みなら何もしない
- 初回のダウンロードは CPU wheel（約 60 MB）で、プロキシ越しでも1分ほど

## テスト

`scripts/run-tests.sh` が `nabla/pjrt/tests` も実行する。プラグインが無ければ
`skip-unless-pjrt` がスキップし、`NABLA_REQUIRE_PJRT` が空でなければ失敗にする
（CI は `NABLA_REQUIRE_PJRT=1`。プラグインは `third_party/pjrt.lock` と
`scripts/fetch-pjrt.sh` のハッシュをキーに `NABLA_PJRT_HOME` をキャッシュする）。

## 版を更新するとき

1. PyPI の JSON API（`https://pypi.org/pypi/<name>/<version>/json`）から x86_64
   manylinux wheel の URL と sha256 を取り、`third_party/pjrt.lock` を直す
   （ダウンロードして `sha256sum` でも確かめる）
2. 新しいプラグインの API の版がヘッダ（0.116）を超えるなら、ヘッダも更新する
3. この文書の表を直す
