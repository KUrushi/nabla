# IREE のソースビルド手順

nabla は IREE を Python ホイールに頼らず、固定したコミットからソースビルドして使う。
このドキュメントはそのビルド手順と、確認した内容の記録。

## 固定したバージョン

`third_party/iree.lock` に記録している。

- リポジトリ: <https://github.com/iree-org/iree>
- タグ: `v3.11.0`
- コミット: `e4a3b0405d7d23554da26403658d0e8c3c5ecf25`

以降の issue（IREE バインディングなど）は、この commit の以下のヘッダを正とする。

- `compiler/bindings/c/iree/compiler/embedding_api.h`
- `runtime/src/iree/runtime/api.h`

## なぜソースビルドか

- ユーザーの方針として、IREE の Python ホイールやリリース配布物 (release assets) を
  流用せず、フルソースビルドを選んだ。そのため release artifacts は試していない。
- Python はランタイム依存にしない（CLAUDE.md）。C API を dlopen して使うため、
  `libIREECompiler.so` と埋め込みランタイムの共有ライブラリを自前でビルドする必要がある。

## 使い方

```sh
# CPU (llvm-cpu) のみ
scripts/build-iree.sh

# CUDA も有効化 (要 CUDA toolkit; 本環境にはない)
scripts/build-iree.sh --cuda
# または
NABLA_IREE_CUDA=1 scripts/build-iree.sh

# cmake の configure だけ行い、ビルドはしない（設定値の確認用）
scripts/build-iree.sh --configure-only

# ビルド後、matmul の StableHLO サンプルをコンパイル・実行して確認する
scripts/verify-iree.sh
scripts/verify-iree.sh --cuda   # nvidia-smi があるときだけ cuda 側も確認する
```

環境変数（すべて省略可能、括弧内は既定値）:

- `NABLA_IREE_SRC` (`/home/user/iree-src`) — IREE のソースチェックアウト先。既に
  ロックしたコミットのチェックアウトならそのまま再利用する。
- `NABLA_IREE_BUILD` (`/home/user/iree-build`) — cmake のビルドディレクトリ。
- `NABLA_IREE_HOME` (`$HOME/.local/share/nabla/iree-3.11.0`) — インストール先
  （`lib/`, `bin/`, `include/`）。nabla の IREE バインディングはここを見る。
- `NABLA_IREE_JOBS` (`nproc`) — ビルドの並列度。
- `NABLA_IREE_CUDA=1` / `--cuda` — CUDA ターゲット・ドライバを有効化する。

## ビルドオプション（cmake）と根拠

`/home/user/iree-src/CMakeLists.txt` の該当行を確認して使っている（行番号はロックした
コミット時点のもの。IREE を更新したら再確認すること）。

| オプション | 値 | 根拠 (CMakeLists.txt) |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | 最適化ビルド。デバッグビルドは今回不要 |
| `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` | `clang` / `clang++` | 環境に用意されている clang 18 を使う |
| `IREE_ENABLE_LLD` | `ON` | L719: `cmake_dependent_option(IREE_ENABLE_LLD ... OFF "NOT APPLE" OFF)`。リンクを高速化 |
| `IREE_ENABLE_THIN_ARCHIVES` | `ON` | L539。ディスクが 27GB しか空いていないため、厚い静的アーカイブを避ける |
| `IREE_ENABLE_WERROR_FLAG` | `OFF` | L541（既定 `ON`）。システムの clang が upstream CI より新しく、`-Werror` で無関係な警告が
  エラーになる可能性があるため |
| `IREE_ENABLE_ASSERTIONS` | `OFF` | L605。Release ビルドでは既定で OFF |
| `IREE_BUILD_COMPILER` | `ON`（既定のまま） | L78。コンパイラ (`libIREECompiler.so`, `iree-compile`) が必要 |
| `IREE_BUILD_TESTS` | `OFF` | L79（既定 `ON`）。IREE 自体のテストはビルドしない |
| `IREE_BUILD_SAMPLES` | `OFF` | L81（既定 `ON`） |
| `IREE_BUILD_PYTHON_BINDINGS` | `OFF` | L82（既定 `OFF`）。Python はランタイム依存にしない |
| `IREE_BUILD_BINDINGS_TFLITE` / `..._JAVA` | `OFF` | L108-109（既定 `ON`）。TFLite 互換シムは不要 |
| `IREE_TARGET_BACKEND_DEFAULTS` | `OFF` | L466。既定で有効になる全ターゲットバックエンドを一旦落とし、必要なものだけ選ぶ |
| `IREE_TARGET_BACKEND_LLVM_CPU` | `ON` | L472 |
| `IREE_TARGET_BACKEND_VMVX` | `OFF` | L469（既定 `ON`、`IREE_BUILD_COMPILER` に従属）。v1 では使わない |
| `IREE_TARGET_BACKEND_CUDA` | `--cuda` 時のみ `ON` | L485。既定は `IREE_TARGET_BACKEND_DEFAULTS` に従うが CUDA toolkit がないと
  自動的に `OFF` になる（L481-483: `IREE_CUDA_AVAILABLE` を見る） |
| `IREE_HAL_DRIVER_DEFAULTS` | `OFF` | L279。既定で有効な全 HAL ドライバを落とす |
| `IREE_HAL_DRIVER_LOCAL_SYNC` / `..._LOCAL_TASK` | `ON` | L318-319。CPU 実行に必要 |
| `IREE_HAL_DRIVER_CUDA` | `--cuda` 時のみ `ON` | L316 |
| `IREE_INPUT_STABLEHLO` | `ON` | L496（既定 `ON`）。nabla は StableHLO しか出力しない |
| `IREE_INPUT_TORCH` / `IREE_INPUT_TOSA` | `OFF` | L497-498（既定 `ON`）。使わない入力方言 |
| `IREE_ERROR_ON_MISSING_SUBMODULES` | `OFF` | L778（既定 `ON`）。`build_tools/scripts/git/check_submodule_init.py` は `IREE_BUILD_COMPILER=ON` のとき ROCm / Vulkan / WebGPU / Torch / tracing 用など、この
  ビルドで使わないサブモジュール（`hip-build-deps` など）まで初期化済みであることを要求してくる。今回は使う7個のサブモジュールだけを意図的に `--depth 1` で取得しているため、このチェックを切る |

ビルドターゲット（`cmake --build ... --target ...`）:

- `iree-compile`, `iree-run-module` — CLI ツール（動作確認・デバッグ用。nabla 本体は
  埋め込み C API を dlopen して呼ぶため、`iree-compile` をサブプロセス起動はしない）
- `iree_compiler_API_SharedImpl` — `compiler/src/iree/compiler/API/CMakeLists.txt` で
  `OUTPUT_NAME "IREECompiler"`, `SOVERSION 0` として定義。`lib/libIREECompiler.so(.0)` を
  生成する
- `iree_runtime_unified` — `runtime/src/iree/runtime/CMakeLists.txt` の
  `iree_cc_unified_library(NAME unified ROOT ::impl)` で定義。ビルドすると
  `runtime/src/iree/runtime/libiree_runtime_unified.a`（静的ライブラリ）ができる

## ランタイムの共有ライブラリ (`libnabla_iree_runtime.so`)

IREE のランタイムは `iree_runtime_unified` として静的アーカイブでしか提供されないため、
CFFI から `dlopen` できるよう、`--whole-archive` で包んで共有ライブラリに変換する。

```sh
clang -shared -fPIC -fuse-ld=lld \
  -o "$NABLA_IREE_HOME/lib/libnabla_iree_runtime.so" \
  -Wl,--whole-archive "$BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a" \
  -Wl,--no-whole-archive \
  -Wl,--no-undefined \
  -lpthread -ldl -lm
```

リンク後、`nm -D` で `iree_runtime_instance_create` と
`iree_hal_driver_registry_default` が export されていることを確認する（されていなければ
ビルドスクリプトはエラーで止まる）。`--no-undefined` でリンクに失敗した場合は、スクリプトが
未解決シンボルの調べ方（`nm -u`）をログに出す。フラグを黙って外すことはしない。

## インストール先のレイアウト (`NABLA_IREE_HOME`)

```
lib/
  libIREECompiler.so, libIREECompiler.so.0   # コンパイラの C API 実装
  libnabla_iree_runtime.so                    # ランタイムのシム共有ライブラリ
  libiree_runtime_unified.a                   # デバッグ用に元の静的アーカイブも残す
bin/
  iree-compile, iree-run-module               # 動作確認用 CLI（本体からは呼ばない）
include/
  iree/compiler/{embedding_api.h, api_support.h, loader.h, mlir_interop.h}
  iree/runtime/, iree/hal/, iree/vm/, ...      # ランタイムのヘッダ一式（ビルドで生成される
                                                 スキーマヘッダも含む）
```

## matmul の StableHLO サンプルでの確認

`tests/fixtures/stablehlo/matmul.mlir` に、`stablehlo.dot_general` を使った
2x3 × 3x2 → 2x2 の行列積を手書きしてある。期待値はコメントに書いてあり、
`scripts/verify-iree.sh` がその値を `iree-run-module` の出力から探して比較する。

`iree-compile` のフラグは v3.x で次の名前を使う（`iree-compile --help` で確認済みの
コード上のオプション名。実行結果自体は下の TODO 表に記録する）:

- `--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu`
  （`compiler/src/iree/compiler/Dialect/HAL/Target/Devices/LocalDevice.cpp` /
  `TargetOptions.cpp` で定義を確認した）
- 実行は `iree-run-module --device=local-task --module=... --function=main --input=...`
- 古い `--iree-hal-target-backends=llvm-cpu` も同じオプションパーサ
  （`TargetOptions.cpp` の `legacyTargetBackends`）で受理されるはずだが、上の新しい
  フラグを正として使う
- CUDA では `--iree-hal-target-device=cuda` と `--device=cuda` を使う
  （`scripts/verify-iree.sh --cuda`。`nvidia-smi` が無ければ自動的にスキップする）

## ツールチェイン（このビルドを検証した環境）

```
$ clang --version
Ubuntu clang version 18.1.3 (1ubuntu1)
$ cmake --version
cmake version 3.28.3
$ ninja --version
1.11.1
$ python3 --version
Python 3.11.15
```

CPU: 4 コア、メモリ 15 GB、GPU なし。

## ディスク容量の注意

このビルド用ファイルシステム（`/`, `/home`, `/tmp` は同一の ext4）の空き容量は
ビルド開始時点で約 27 GB しかない。IREE + LLVM のフルソースビルドはこれを大きく
消費しうるため、`IREE_ENABLE_THIN_ARCHIVES=ON` にして厚いアーカイブを避け、
`IREE_BUILD_TESTS` / `IREE_BUILD_SAMPLES` など不要なものを OFF にしている。
`scripts/build-iree.sh` はビルド前後で `df -h` と `du -sh` をログに出す。

## GPU / CUDA について

この開発環境には NVIDIA GPU がなく、CUDA toolkit もインストールされていない
（`IREE_CUDA_AVAILABLE` が偽になり、`IREE_TARGET_BACKEND_CUDA` は既定でも自動的に
OFF になる。`CMakeLists.txt` L481-483）。そのため:

- `scripts/build-iree.sh --cuda` のコード自体は用意したが、この環境では実行できず
  未検証（CUDA toolkit が必要）
- `scripts/verify-iree.sh --cuda` は `nvidia-smi` の有無を見て、無ければ CPU 側の
  検証だけ行いスキップする
- `--iree-cuda-target=sm_XX` の指定や、LLVM がその世代に対応しているかの確認は、
  GPU を持つ環境が用意でき次第行う

## TODO（オーケストレーターが実行結果を埋める）

このユニット（実装担当）はフルビルドを実行していない（オーケストレーターが実行する
分担のため）。`bash -n` での構文チェックと、`scripts/build-iree.sh --configure-only`
（一時ディレクトリに configure するだけ、実行後に削除）は実施し、以下を確認した。

- cmake configure が約 35 秒で成功し、`cmake -L -N` で表の全オプションが意図通りの値
  （`IREE_TARGET_BACKEND_LLVM_CPU=ON` / `..._CUDA=OFF`、`IREE_HAL_DRIVER_LOCAL_SYNC=ON`
  / `..._LOCAL_TASK=ON` / `..._CUDA=OFF`、`IREE_INPUT_STABLEHLO=ON` /
  `..._TORCH=OFF` / `..._TOSA=OFF`、`IREE_BUILD_TESTS=OFF` / `..._SAMPLES=OFF` /
  `..._PYTHON_BINDINGS=OFF`）になっていることを確認した
- `IREE_BUILD_COMPILER=ON` のとき、IREE 側の `check_submodule_init.py` が
  今回意図的に初期化していないサブモジュール（`hip-build-deps` など ROCm/Vulkan/
  WebGPU/Torch/tracing 用）まで要求してくることが分かったため、
  `-DIREE_ERROR_ON_MISSING_SUBMODULES=OFF` を追加した（上の表に理由を記載）

| 項目 | 結果 | 所要時間 | 備考 |
| --- | --- | --- | --- |
| `scripts/build-iree.sh` (llvm-cpu) フルビルド | 未実施 | - | オーケストレーターが実行 |
| `scripts/verify-iree.sh` (llvm-cpu) | 未実施 | - | 上のビルド完了後に実行 |
| `scripts/build-iree.sh --cuda` フルビルド | 未実施（環境に GPU/CUDA なし） | - | CUDA toolkit のある環境が必要 |
| `scripts/verify-iree.sh --cuda` | 未実施（環境に GPU/CUDA なし） | - | 同上 |
| リリース配布物 (release artifacts) の動作確認 | 未実施 | - | ユーザーの方針でフルソースビルドを選択したため試していない |
