# IREE のビルド手順

nabla は IREE を固定したコミットで使う。このドキュメントはその手順と、実際にこの環境
（4 コア / 15 GB RAM、GPU なし）で確認した内容の記録。

## 固定したバージョン

`third_party/iree.lock` に記録している。

- リポジトリ: <https://github.com/iree-org/iree>
- タグ: `v3.11.0`
- コミット: `e4a3b0405d7d23554da26403658d0e8c3c5ecf25`
- コンパイラ用ホイール（後述）: `iree-base-compiler==3.11.0`
  （`iree_base_compiler-3.11.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
  sha256 `ac3505591b6b134784eae7bcdf806fc66a2d120a82134b98bd4fbe488fdf84c5`）

以降の issue（IREE バインディングなど）は、この commit の以下のヘッダを正とする。

- `compiler/bindings/c/iree/compiler/embedding_api.h`
- `runtime/src/iree/runtime/api.h`

## 環境変数

`scripts/build-iree.sh` が読む環境変数（すべて省略時は既定値を使う）。

| 変数 | 既定値 | 意味 |
| --- | --- | --- |
| `NABLA_IREE_HOME` | `~/.local/share/nabla/iree-3.11.0` | インストール先。ビルド成果物 (`lib/`, `bin/`, `include/`) をここに置く |
| `NABLA_IREE_SRC` | `${XDG_CACHE_HOME:-~/.cache}/nabla/iree-src` | IREE のソースチェックアウト先（wheel モードでも C API ヘッダとランタイムのソースに使う） |
| `NABLA_IREE_BUILD` | wheel: `${XDG_CACHE_HOME:-~/.cache}/nabla/iree-build-runtime`<br>source: `${XDG_CACHE_HOME:-~/.cache}/nabla/iree-build` | cmake のビルドディレクトリ |
| `NABLA_IREE_WHEEL_DIR` | `${XDG_CACHE_HOME:-~/.cache}/nabla/iree-wheel` | ダウンロードしたホイールと展開先のキャッシュ |
| `NABLA_IREE_COMPILER` | `wheel` | `wheel` または `source`。`--compiler` と同じ |
| `NABLA_IREE_CUDA` | `0` | `1` で CUDA を有効化。`--cuda` と同じ |
| `NABLA_IREE_JOBS` | `$(nproc)` | `ninja` の並列数 |

キャッシュ用のディレクトリ (`NABLA_IREE_SRC` / `NABLA_IREE_BUILD` / `NABLA_IREE_WHEEL_DIR`)
は [XDG Base Directory](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
の慣習に従い、既定では `$XDG_CACHE_HOME`（未設定なら `~/.cache`）の下の `nabla/` に置く。
どの変数も明示的に指定すればそちらを使うので、固定のパスを既に使っている開発環境や
CI があれば、その環境変数を設定するだけで既定値を変えずに済む（例えばこの既定値を
導入する前から `NABLA_IREE_SRC=/home/user/iree-src` のようなパスを使っていた環境は、
その変数を設定し続ける限りディレクトリを移動する必要はない）。

`NABLA_IREE_SRC` が存在しない場合、`scripts/build-iree.sh` はロックしたタグ・コミットを
`git clone --depth 1` してから必要なサブモジュールだけを浅く取得する（詳細は次の節）。

## コンパイラの入手方法: wheel（既定）と source

`scripts/build-iree.sh` は `--compiler=wheel|source`（環境変数 `NABLA_IREE_COMPILER`）で
コンパイラ (`libIREECompiler.so`, `iree-compile`) の入手方法を選べる。**既定は
`wheel`**。ランタイム (`libnabla_iree_runtime.so`) は、どちらのモードでも常にロックした
コミットからソースビルドする（PyPI のホイールには C ランタイムのライブラリは含まれて
いないため）。

**wheel モードは x86_64 Linux 専用**（`third_party/iree.lock` に記録したホイールが
`manylinux_2_28_x86_64` cp311 タグのみだから）。macOS や aarch64 など、それ以外の
プラットフォームでは `--compiler=source`（フルソースビルド）を使う。

### なぜ既定が wheel なのか（このマシンでの実測）

当初の方針は Python の配布物に頼らないフルソースビルドだったが、ユーザーからは
「フルソースビルドが1時間を超えたら PyPI パッケージを使ってよい」という条件が付いていた。
この環境 (4 コア / 15 GB RAM、GPU なし) で実際にフルビルド（Release、clang 18 + lld、
llvm-cpu のみ、local-sync/local-task、StableHLO 入力のみ、tests/samples/python は無効）を
2026-09-26 05:28 UTC に開始したところ:

- ランタイム側のターゲットは約10秒で終わった
- コンパイラ側は 59 分経過した時点で ninja のステップが 7085 中 5674 までしか進まず、
  そこで打ち切った

そのため、**コンパイラは既定で third_party/iree.lock に記録した PyPI ホイール
(`iree-base-compiler==3.11.0`) を使う**ことにした。このホイールの
`iree-compile --version` は

```
IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25
```

を報告し、ロックしたコミットと同じビルドであることを確認できる。`scripts/build-iree.sh`
はこの出力にロックしたコミットのハッシュが含まれることを検証してから使う。

Python はこのダウンロード・展開（`pip download` と `python -m zipfile`）にしか使わず、
実行時には使わない（CLAUDE.md の方針どおり）。`iree-compile` をサブプロセスとして
呼ぶのも、この検証と `scripts/verify-iree.sh` のような動作確認のためだけであり、nabla
本体は埋め込み C API を dlopen して呼ぶ。

フルソースビルド (`--compiler=source`) のコードパスは残してあり、CI やより強力な
マシンで使える。ただしこの環境では最後まで実行できておらず未検証（下の TODO 表参照）。

### wheel から取り出すファイル

ホイール (`.whl`、実体は zip) の中の `iree/compiler/_mlir_libs/` に、コンパイラの
C API 実装と CLI が入っている。

```
iree/compiler/_mlir_libs/libIREECompiler.so   # → lib/libIREECompiler.so
iree/compiler/_mlir_libs/iree-compile         # → bin/iree-compile
iree/compiler/_mlir_libs/iree-lld             # → bin/iree-lld
```

`iree-lld` はロックしたコミットにバンドルされている LLD（IREE 自前ビルドの busybox
版）。**llvm-cpu バックエンドは、コンパイル済みの CPU 実行ファイルを常に外部のリンカを
サブプロセスとして起動してリンクする**（このビルドにインプロセスのリンカは無い）ため、
`iree-compile` はリンカを何らかの方法で見つける必要がある。デフォルトでは
`--iree-llvmcpu-embedded-linker-path` を指定しない限り `PATH` 上のリンカ（この環境では
システムの `/usr/bin/lld`）を拾ってしまい、ロックした IREE コミットのビルドに使われた
LLD のバージョンとずれる可能性がある（LLD はリンクするオブジェクトの ABI に敏感な
ツールで、IREE 側は特定バージョンとの組み合わせでテストされている）。そのため
`scripts/verify-iree.sh`（および `#20` のより上位のビルド層）は、`$NABLA_IREE_HOME/bin/iree-lld`
が存在すればそれを `--iree-llvmcpu-embedded-linker-path` に明示的に渡し、システムの
リンカではなくロックしたコミットと同じビルドの LLD を必ず使わせる。

`iree-compile` と同様、`iree-lld` の RUNPATH も `$ORIGIN` で同じディレクトリの
`libIREECompiler.so` を必要とするため、`bin/libIREECompiler.so` へのシンボリックリンク
（既存）がそのまま効く。Python の `zipfile` が実行ビットを復元しない点も `iree-compile`
と同じで、`chmod +x` で復元する。`scripts/build-iree.sh` はインストール直後に
`iree-lld -flavor gnu --version` を実行し、起動して `libIREECompiler.so` を解決できる
ことを確認する（`iree-lld` は引数なしだと使い方を表示するだけで終了コード非ゼロになる
ため、`-flavor gnu --version` で実際にサブコマンドとして動かす）。

コンパイラの C API ヘッダ（`embedding_api.h` 等）はホイールには含まれていないため、
どちらのモードでも `NABLA_IREE_SRC` のソースチェックアウトからコピーする。

注意点（`scripts/build-iree.sh` が対処している）:

- `iree-compile` の RUNPATH は `$ORIGIN` で、同じディレクトリの `libIREECompiler.so` を
  必要とする。そのためインストール先では `bin/libIREECompiler.so` という
  シンボリックリンクを `lib/libIREECompiler.so` に張っている
- Python の `zipfile` モジュールはアーカイブに記録された Unix の実行ビットを復元しない
  ため、展開直後の `iree-compile` は `-rw-r--r--` になる。`chmod +x` で復元してから使う
- ダウンロードしたホイールは third_party/iree.lock の sha256 と照合してから使う
  （一致しなければエラーで止まる）
- ホイールのダウンロードと sha256 検証は、cmake configure の**後**・ninja による
  ランタイムのビルドより**前**に行う（`--configure-only` は cmake configure の
  直後に止まるので、ホイールには一切触れない）。壊れたホイールや sha256 の
  不一致は、ランタイムのビルドを始める前に即座に検出して止まるので、ビルド
  ディレクトリやインストール先に半端な状態を残さない
- `libIREECompiler.so`（wheel モードで約 337 MB）は、インストール先に既に同じ
  サイズ・mtime のファイルがあれば再コピーしない（`copy_if_changed`、
  `--compiler=source` でビルドツリーからコピーする場合も同様）。展開・ビルド
  結果を毎回同じ手順で作る限り、`cp -a` はソースの mtime を保存するので、
  2回目以降の実行はこの比較だけで済み、大きなファイルの無駄な再コピーを
  避けられる

## ビルドオプション（cmake）と根拠

`/home/user/iree-src/CMakeLists.txt` の該当行を確認して使っている（行番号はロックした
コミット時点のもの。IREE を更新したら再確認すること）。

両モード共通:

| オプション | 値 | 根拠 (CMakeLists.txt) |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | 最適化ビルド。デバッグビルドは今回不要 |
| `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` | `clang` / `clang++` | 環境に用意されている clang 18 を使う |
| `IREE_ENABLE_LLD` | `ON` | L719: `cmake_dependent_option(IREE_ENABLE_LLD ... OFF "NOT APPLE" OFF)`。リンクを高速化 |
| `IREE_ENABLE_THIN_ARCHIVES` | `ON` | L539。ディスクが乏しいため、厚い静的アーカイブを避ける |
| `IREE_ENABLE_WERROR_FLAG` | `OFF` | L541（既定 `ON`）。システムの clang が upstream CI より新しく、`-Werror` で無関係な警告が
  エラーになる可能性があるため |
| `IREE_ENABLE_ASSERTIONS` | `OFF` | L605。Release ビルドでは既定で OFF |
| `IREE_VISIBILITY_HIDDEN` | `OFF` | L133（既定 `ON`）。既定では全ライブラリが `-fvisibility=hidden`
  でビルドされ（`iree_copts.cmake`）、C の `IREE_API_EXPORT` も何も付与しない
  (`runtime/src/iree/base/attributes.h`) ため、`libnabla_iree_runtime.so` から
  `iree_*` 関数が一切 export されず CFFI の `dlopen`/`dlsym` が失敗する。可視性は
  リンク時のフラグでは覆せないため、configure 時に明示的に `OFF` にする必要がある |
| `IREE_BUILD_TESTS` | `OFF` | L79（既定 `ON`）。IREE 自体のテストはビルドしない |
| `IREE_BUILD_SAMPLES` | `OFF` | L81（既定 `ON`） |
| `IREE_BUILD_PYTHON_BINDINGS` | `OFF` | L82（既定 `OFF`）。Python はランタイム依存にしない |
| `IREE_BUILD_BINDINGS_TFLITE` / `..._JAVA` | `OFF` | L108-109（既定 `ON`）。TFLite 互換シムは不要 |
| `IREE_ERROR_ON_MISSING_SUBMODULES` | `OFF` | L778（既定 `ON`）。`build_tools/scripts/git/check_submodule_init.py` は `IREE_BUILD_COMPILER=ON` のとき ROCm / Vulkan / WebGPU / Torch / tracing 用など、この
  ビルドで使わないサブモジュール（`hip-build-deps` など）まで初期化済みであることを要求してくる。今回は使う7個のサブモジュールだけを意図的に `--depth 1` で取得しているため、このチェックを切る |
| `IREE_HAL_DRIVER_DEFAULTS` | `OFF` | L279。既定で有効な全 HAL ドライバを落とす |
| `IREE_HAL_DRIVER_LOCAL_SYNC` / `..._LOCAL_TASK` | `ON` | L318-319。CPU 実行に必要 |
| `IREE_HAL_DRIVER_CUDA` | `--cuda` 時のみ `ON` | L316 |

`--compiler=wheel` 時のみ:

| オプション | 値 | 根拠 |
| --- | --- | --- |
| `IREE_BUILD_COMPILER` | `OFF` | コンパイラは PyPI ホイールから取るので、このビルドは
  ランタイムだけを作る。`IREE_TARGET_BACKEND_*` / `IREE_INPUT_*` は
  `cmake_dependent_option(... ${IREE_BUILD_COMPILER} OFF)` の形で `IREE_BUILD_COMPILER`
  に従属しているため、これが `OFF` なら自動的にすべて `OFF` になり、明示的に指定する
  必要はない |

`--compiler=source` 時のみ（既定は `--cuda` なしで CUDA 側は `OFF`）:

| オプション | 値 | 根拠 |
| --- | --- | --- |
| `IREE_BUILD_COMPILER` | `ON`（既定のまま） | L78。コンパイラ (`libIREECompiler.so`, `iree-compile`) が必要 |
| `IREE_TARGET_BACKEND_DEFAULTS` | `OFF` | L466。既定で有効になる全ターゲットバックエンドを一旦落とし、必要なものだけ選ぶ |
| `IREE_TARGET_BACKEND_LLVM_CPU` | `ON` | L472 |
| `IREE_TARGET_BACKEND_VMVX` | `OFF` | L469（既定 `ON`、`IREE_BUILD_COMPILER` に従属）。v1 では使わない |
| `IREE_TARGET_BACKEND_CUDA` | `--cuda` 時のみ `ON` | L485。既定は `IREE_TARGET_BACKEND_DEFAULTS` に従うが CUDA toolkit がないと
  自動的に `OFF` になる（L481-483: `IREE_CUDA_AVAILABLE` を見る） |
| `IREE_INPUT_STABLEHLO` | `ON` | L496（既定 `ON`）。nabla は StableHLO しか出力しない |
| `IREE_INPUT_TORCH` / `IREE_INPUT_TOSA` | `OFF` | L497-498（既定 `ON`）。使わない入力方言 |

ビルドターゲット（`cmake --build ... --target ...`）:

- `iree-run-module`, `iree_runtime_unified` — 両モード共通。`iree-run-module` は
  動作確認・デバッグ用の CLI（本体からは呼ばない）。`iree_runtime_unified` は
  `runtime/src/iree/runtime/CMakeLists.txt` の `iree_cc_unified_library(NAME unified
  ROOT ::impl)` で定義され、ビルドすると
  `runtime/src/iree/runtime/libiree_runtime_unified.a`（静的ライブラリ）ができる
- `iree-compile`, `iree_compiler_API_SharedImpl`, `iree-lld` — `--compiler=source` のとき
  だけ追加でビルドする。`iree_compiler_API_SharedImpl` は
  `compiler/src/iree/compiler/API/CMakeLists.txt` で `OUTPUT_NAME "IREECompiler"`,
  `SOVERSION 0` として定義され、`lib/libIREECompiler.so(.0)` を生成する。`iree-lld` は
  `tools/CMakeLists.txt` で `IREE_ENABLE_LLD=ON`（既定で設定済み）のときだけ定義される
  ターゲットで、`iree-compile` が `--iree-llvmcpu-embedded-linker-path` に渡す
  リンカ本体を提供する（`ninja -t targets all | grep -i lld` で確認）

## ランタイムの共有ライブラリ (`libnabla_iree_runtime.so`)

IREE のランタイムは `iree_runtime_unified` として静的アーカイブでしか提供されないため、
CFFI から `dlopen` できるよう、`--whole-archive` で包んで共有ライブラリに変換する。
これはどちらのコンパイラモードでも同じ手順。

`iree_cc_unified_library`（`iree_runtime_unified` の定義に使われるマクロ、
`build_tools/cmake/iree_cc_library.cmake`）は third_party の依存（flatcc, printf）を
アーカイブの中に含めず `INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS` として記録するだけなので、
`libiree_runtime_unified.a` 単体をリンクすると `flatcc_verify_table_as_root` や
`vsnprintf_` / `vfctprintf` が未定義シンボルになる。ビルドツリー内の
`libflatcc_parsing.a` と `libprintf_printf.a` を探して一緒にリンクする必要がある。

```sh
clang -shared -fPIC -fuse-ld=lld \
  -o "$NABLA_IREE_HOME/lib/libnabla_iree_runtime.so" \
  -Wl,--whole-archive \
    "$BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a" \
    "$BUILD/build_tools/third_party/flatcc/libflatcc_parsing.a" \
    "$BUILD/build_tools/third_party/printf/libprintf_printf.a" \
  -Wl,--no-whole-archive \
  -Wl,--no-undefined \
  -lpthread -ldl -lm
```

（`scripts/build-iree.sh` はこの2つのパスを固定せず、ビルドツリーを `find` して探す。
CMake のバージョンやターゲット構成が変わってサブディレクトリが動いても壊れないように
するため）

リンク後、`nm -D` で `iree_runtime_instance_create` と
`iree_hal_driver_registry_default` が export されていることを確認する（されていなければ
ビルドスクリプトはエラーで止まる）。`--no-undefined` でリンクに失敗した場合は、スクリプトが
未解決シンボルの調べ方（`nm -u`）をログに出す。フラグを黙って外すことはしない。

この検証は `nm -D "$SHIM_SO" | grep -qE ...` という素朴なパイプでは書けない
（`set -o pipefail` の下では、`grep -q` が最初のマッチで先に終了して `nm` に
`SIGPIPE` が飛び、`nm` がマッチしていても非ゼロ終了することでパイプライン全体が
失敗扱いになることを実際にこの環境で確認した）。そのため `nm -D` の出力を一度変数に
captureしてから `grep` する形にしている。

## コンパイラのインストール先レイアウト (`NABLA_IREE_HOME`)

```
lib/
  libIREECompiler.so[.0]                      # コンパイラの C API 実装
                                               # (wheel モードでは .0 サフィックスなし)
  libnabla_iree_runtime.so                    # ランタイムのシム共有ライブラリ
bin/
  iree-compile, iree-run-module               # 動作確認用 CLI（本体からは呼ばない）
  iree-lld                                    # llvm-cpu が --iree-llvmcpu-embedded-
                                               # linker-path で使う、ロックしたコミットと
                                               # 同じビルドの LLD
  libIREECompiler.so -> ../lib/libIREECompiler.so  # wheel モードのみ。iree-compile /
                                               # iree-lld の RUNPATH=$ORIGIN が要求する
include/
  iree/compiler/{embedding_api.h, api_support.h, loader.h, mlir_interop.h}
  iree/runtime/, iree/hal/, iree/vm/, ...      # ランタイムのヘッダ一式（ビルドで生成される
                                                 スキーマヘッダも含む）
```

サイズの実測値（wheel モード、CPU のみ）:

- `lib/libIREECompiler.so`: 約 337 MB
- `lib/libnabla_iree_runtime.so`: 約 1 MB（1380 個の `iree_*` シンボルを export）
- `NABLA_IREE_HOME` 全体: 約 345 MB

## matmul の StableHLO サンプルでの確認

`tests/fixtures/stablehlo/matmul.mlir` に、`stablehlo.dot_general` を使った
2x3 × 3x2 → 2x2 の行列積を手書きしてある。期待値はコメントに書いてあり、
`scripts/verify-iree.sh` がその値を `iree-run-module` の出力から探して比較する。

`iree-compile` のフラグは v3.x で次の名前を使う（`compiler/src/iree/compiler/Dialect/HAL/Target/Devices/LocalDevice.cpp` / `TargetOptions.cpp` で定義を確認した）:

- `--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu`
- `--iree-llvmcpu-target-cpu=host` — 省略するとジェネリックな CPU 世代向けにコンパイル
  され、`iree-compile` が警告を出す。このマシンの CPU を指定して黙らせる
- 実行は `iree-run-module --device=local-task --module=... --function=main --input=...`
- 古い `--iree-hal-target-backends=llvm-cpu` も同じオプションパーサ
  （`TargetOptions.cpp` の `legacyTargetBackends`）で受理されるはずだが、上の新しい
  フラグを正として使う
- CUDA では `--iree-hal-target-device=cuda` と `--device=cuda` を使う
  （`scripts/verify-iree.sh --cuda`。`nvidia-smi` が無ければ自動的にスキップする）

### 実行結果（この環境、wheel モード、llvm-cpu、2026-09-26）

```
$ NABLA_IREE_HOME=/tmp/xxx/iree scripts/build-iree.sh --compiler=wheel
...
[build-iree] ... confirmed exports: iree_runtime_instance_create, iree_hal_driver_registry_default
[build-iree] ... confirmed: iree-compile reports commit e4a3b0405d7d23554da26403658d0e8c3c5ecf25
[build-iree] ... install complete: /tmp/xxx/iree
[build-iree] ... elapsed: 6s

$ NABLA_IREE_HOME=/tmp/xxx/iree scripts/verify-iree.sh
[verify-iree] compiling for llvm-cpu -> .../nabla-matmul-llvm-cpu.vmfb
[verify-iree] running llvm-cpu module with iree-run-module
[verify-iree] llvm-cpu: OK, output matches expected [[58 64][139 154]] exactly
[verify-iree] skipping cuda check (pass --cuda to enable; requires an IREE build with IREE_TARGET_BACKEND_CUDA=ON)
[verify-iree] all checks passed
```

`iree-run-module` の実際の出力は `2x2xf32=[58 64][139 154]`（[[1,2,3],[4,5,6]] と
[[7,8],[9,10],[11,12]] の積として正しい）。

### verify-iree.sh の判定バグの修正

旧版の `check_output` は `\b58\b.*\b64\b.*\b139\b.*\b154\b` という正規表現で「4つの数値が
この順に現れるか」だけを見ていた。しかし `\b`（単語境界）は `58.5` のような小数の前でも
成立する（`58.5` の直前・"58" と "." の間は単語境界になる）ため、この正規表現は
`58.5 64 139 154` のような**間違った**（小数が混じった）結果にもマッチしてしまっていた。
これは実際に再現・確認済み（fixtures を使わない単体テストで `58.5` を含む出力を作って
`check_output` に渡すと、旧正規表現は誤って通していた）。

修正後は、`iree-run-module` の出力からテンソルの中身（バッファビュー行の最後の `=`
より後ろ）だけを取り出し、そこに現れる数値の並びが `58 64 139 154` と**完全に一致する
か**を比較する。これにより小数が混じった誤った結果も、余分な数値が混じった結果も
確実に弾く。

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

このビルド用ファイルシステム（`/`, `/home`, `/tmp` は同一）の空き容量は開始時点で
約 24–27 GB。IREE + LLVM のフルソースビルド（`--compiler=source`）はこれを大きく
消費しうるため、`IREE_ENABLE_THIN_ARCHIVES=ON` にして厚いアーカイブを避け、
`IREE_BUILD_TESTS` / `IREE_BUILD_SAMPLES` など不要なものを OFF にしている。wheel モード
（既定）はコンパイラのフルビルドをしないため、ビルドディレクトリは数十 MB、
`NABLA_IREE_HOME` は約 345 MB で済む。`scripts/build-iree.sh` はビルド前後で `df -h` と
`du -sh` をログに出す。

## GPU / CUDA について

この開発環境には NVIDIA GPU がなく、CUDA toolkit もインストールされていない
（`IREE_CUDA_AVAILABLE` が偽になる）。そのため CUDA 関連は**未検証**:

`--cuda`（`--compiler=source` と組み合わせた場合）は、CUDA toolkit がローカルに
インストール済みでなければ、cmake configure 時に NVIDIA の redistributable
パッケージ索引にネットワークで到達してターゲットの依存関係を取得しようとする。
この環境ではプロキシ経由のアクセスが `403 Forbidden` で拒否され、configure の
段階で失敗した（GPU の有無以前に、ネットワークアクセスの問題）。CUDA toolkit を
インストールするか、NVIDIA の索引に到達できるネットワークが必要。

- `scripts/build-iree.sh --cuda` は `--compiler=source` と組み合わせたときのみ
  `IREE_TARGET_BACKEND_CUDA=ON` を追加するコードだが、この環境では実行できず未検証
  （CUDA toolkit が必要）。`--compiler=wheel`（既定）と `--cuda` を組み合わせた場合は
  `IREE_HAL_DRIVER_CUDA=ON` でランタイムの CUDA HAL ドライバはビルドされるが、
  コンパイラ側の CUDA ターゲットバックエンドは PyPI ホイールの `iree-base-compiler`
  にどのターゲットが含まれるか未確認であり、これも未検証
- `scripts/verify-iree.sh --cuda` は `nvidia-smi` の有無を見て、無ければ CPU 側の
  検証だけ行いスキップする（この環境では常にスキップされた）。スキップ時の
  メッセージには、CUDA を有効にするには CUDA toolkit のインストールか NVIDIA の
  redistributable 索引へのネットワーク到達性が必要である旨も表示する
- `--iree-cuda-target=sm_XX` の指定や、LLVM がその世代に対応しているかの確認は、
  GPU を持つ環境が用意でき次第行う

## issue #3 の完了条件との対応（正直な報告）

- [x] IREE を固定コミットからビルドする手順がある、かつ実際にこの環境でビルドできる
  — ただし**コンパイラは既定で PyPI ホイールを使う**（フルソースビルドは1時間で
  終わらなかったため、ユーザーの事前の指示に従いホイールに切り替えた）。ランタイムは
  常にソースビルドする。フルソースビルド (`--compiler=source`) のコードパス自体は
  用意したが、この環境では最後まで実行できておらず未検証
- [x] `libIREECompiler.so` とランタイムの共有ライブラリができ、`iree-compile` /
  `iree-run-module` で matmul の StableHLO サンプルが CPU (llvm-cpu / local-task) で
  正しく動くことを確認した（このドキュメントの実行結果を参照）
- [ ] GPU (CUDA) での動作確認 — この環境に GPU が無いため**未検証**。CUDA toolkit と
  GPU がある環境で `--cuda` 付きで再実行して確認する必要がある
- [x] 手順をドキュメント化した（このファイル）

## TODO

| 項目 | 結果 | 所要時間 | 備考 |
| --- | --- | --- | --- |
| `scripts/build-iree.sh --compiler=wheel` (llvm-cpu) | 成功 | 約6秒（ランタイムの
  ニンジャビルドがキャッシュ済みの場合。フルにビルドし直す場合は runtime のビルドに
  約10秒） | このドキュメントの実行結果を参照 |
| `scripts/verify-iree.sh` (llvm-cpu) | 成功 | 数秒 | 上のビルド完了後に実行、
  [[58 64][139 154]] と完全一致することを確認 |
| `scripts/build-iree.sh --compiler=source` フルビルド | 未完了 | 59分経過時点で
  ninja 7085 ステップ中 5674 まで（打ち切り） | このマシンのスペック
  (4 コア / 15 GB RAM) では実用的な時間で終わらない。CI やより強力なマシンで再挑戦。
  `iree-lld` ターゲット自体は cmake configure まで進めて `ninja -t targets all` で
  存在を確認済みだが、ビルドして `bin/iree-lld` の動作確認をするところまでは
  実行できていない（同上の理由） |
| `scripts/build-iree.sh --cuda` | 未実施（環境に GPU/CUDA なし） | - | CUDA toolkit の
  あるマシンが必要 |
| `scripts/verify-iree.sh --cuda` | 未実施（環境に GPU/CUDA なし） | - | 同上 |

## IREE を上げたときに回避策を外せるか確かめる手順

IREE 3.11.0 のコンパイラのバグ3件（issue #73、#165。詳細・最小の再現・上流への報告の下書きは [`docs/iree-upstream-bugs.md`](iree-upstream-bugs.md)）に対して、nabla は回避策を入れている。`third_party/iree.lock` を上げる PR では、次の手順で各バグが直ったかを確かめ、直ったものだけ回避策を外す（外すのは別の PR に分けてよい）。

### 1. 単体の iree-compile で最小の再現を流す

```sh
NABLA_IREE_HOME=<新しい版のインストール先> scripts/check-iree-repros.sh --runs 20
```

`docs/iree-repros/` の各ファイルを `--runs` 回コンパイルし、1行ずつ判定を出す。終了コードは、記録どおり（バグは全部再現し、回避策の形は全部通る）なら 0、どれか違えば 1。IREE 3.11.0 での出力:

```
bug      dot-general-k0.mlir                      REPRODUCED      crashes=20/20 exit-codes=136
bug      while-constant-carry.mlir                REPRODUCED      crashes=19/20 exit-codes=139 0
control  while-constant-carry-barrier.mlir        OK              crashes=0/20 exit-codes=0
bug      while-i1-carry-returned.mlir             REPRODUCED      crashes=20/20 exit-codes=134 139
control  while-i1-carry-not-returned.mlir         OK              crashes=0/20 exit-codes=0
```

`while-constant-carry.mlir` は非決定的に落ちる（3.11.0 では 50 回中 49 回）ので、`NOT-REPRODUCED` を「直った」と判断するのは `--runs 50` 以上で 1 回も落ちなかったときにする。`bug` の行が `NOT-REPRODUCED` になったら、上流の issue（`docs/iree-upstream-bugs.md` の各節）が閉じているかも確かめる。

### 2. ガードテストを流す

```sh
NABLA_IREE_HOME=<新しい版> NABLA_REQUIRE_IREE=1 scripts/run-tests.sh
```

次のテスト（`nabla/iree/tests` の medium）は、**バグが IREE に残っていることを確かめる**ので、バグが直ると失敗する。失敗したら、それが「直った」という合図で、下の撤去条件に従って回避策とテストを一緒に消す。

| バグ | ガードテスト（`tests/iree/`） | 直ったときの結果 |
| --- | --- | --- |
| 1. K=0 の dot_general | ガードテストは無い。`float-traps-test.lisp` の `float-traps/zero-size-dot-general-poisons-compiler-then-fails-clearly` は、K=0 がコンパイルできた場合（`K0-RESULT=COMPILED-OK`）も通るように書いてある | 失敗しない。手順 1 の `dot-general-k0.mlir` で判断する |
| 2. 定数の carry の while | `while-loop-test.lisp` の `while-loop/iree-known-bug-constant-carry-crashes-the-compiler`（子プロセスで 10 回コンパイルし、1 回でも落ちることを確かめる） | 失敗する |
| 3. `i1` の carry を返す while | `while-loop-test.lisp` の `while-loop/iree-known-limitation-i1-carry-returned-crashes-the-compiler` | 失敗する |

回避策が新しい版でも効いていることは、`while-loop/iree-constant-carry-with-barrier-compiles-repeatedly`・`tests/iree/scan-test.lisp`・`tests/iree/dot-test.lisp` の `dot-general/zero-contracting-compiles-and-matches-eager` が引き続き確かめる（こちらは失敗してはいけない）。

### 3. 回避策を撤去する条件と、消すもの

**バグ 1（K=0 の dot_general、#62 / #73）**: 条件は `dot-general-k0.mlir` が `NOT-REPRODUCED`（決定的に落ちていたので `--runs 20` で十分）で、かつ K=0 の dot_general を IREE で実行した結果が全 0 になること。消すもの:

- `src/primitives/dot.lisp` の `%dot-zero-contracting-p` と `%dot-zero-constant-line`、`%dot-emit-lines` の K=0 の分岐（と docstring の issue #62 の記述）
- `tests/primitives/dot-test.lisp` の「K=0（issue #62）: ゼロ定数 :emit」の節のテスト（emit がゼロ定数になることを確かめているので、dot_general を出すことを確かめる形に書き換える）。`tests/iree/dot-test.lisp` の `dot-general/zero-contracting-compiles-and-matches-eager` は残す（回避策を消した後も、K=0 が IREE で正しく動くことを守る）
- `docs/stablehlo-ops.md` の dot_general の行の K=0 の記述
- #69 のコンパイラの poison（`src/iree/compiler.lisp` の `*compiler-poison-reason*` まわり）は、K=0 に限らず「Pipeline の途中で ARITHMETIC-ERROR が起きたら以後そのプロセスでコンパイラを使わない」一般の安全策なので**残す**。ただし、その経路を踏ませるテスト（`tests/iree/float-traps-test.lisp` の性質2）は K=0 では #DE が起きなくなり、`COMPILED-OK` の側しか通らなくなる。別の #DE の起こし方が無ければ、その旨をテストのコメントに書く

**バグ 2（定数の carry の while、#165）**: 条件は `while-constant-carry.mlir` が `--runs 50` で 0 回、かつガードテスト `while-loop/iree-known-bug-constant-carry-crashes-the-compiler` が失敗すること。消すもの:

- `src/while-loop.lisp` の `%while-barrier-lines` と、`while-loop` の `:emit` からの呼び出し・冒頭のコメントのこのバグの記述
- `src/scan.lisp` の `:emit` で、カウンタと ys の0初期値を `stablehlo.optimization_barrier` に通している部分（長さ 1 で barrier を付けない分岐も一緒に）
- `tests/iree/while-loop-test.lisp` の `*wl-iree-const-carry-crash-text*` とガードテスト。`while-loop/iree-constant-carry-with-barrier-compiles-repeatedly` は、`optimization_barrier` を含むことの検査を外して「定数の carry の while が 10 回続けてコンパイルできる」テストとして残す
- `docs/stablehlo-ops.md` の制御構造の節と scan の節のこのバグの記述、`CLAUDE.md` の「設計上の約束」の IREE 3.11 の `stablehlo.while` の項目（barrier の部分）、`docs/phase3-report.md` §3.12 の 1 に「何版で直った」を追記

**バグ 3（`i1` の carry を返す while、#131 / #165）**: 条件は `while-i1-carry-returned.mlir` が `NOT-REPRODUCED`、かつガードテスト `while-loop/iree-known-limitation-i1-carry-returned-crashes-the-compiler` が失敗すること。回避策のコードは無いので、消すのは制限の記述とガードテスト:

- `src/while-loop.lisp` の冒頭の「既知の制限」と `while-loop` の docstring の制限の記述
- `tests/iree/while-loop-test.lisp` の `*wl-iree-known-crash-text*` とガードテスト。代わりに `%with-wl-iree-i1-check` で `i1` の carry を返す形（`return-flag` が真）の、eager との一致のテストを足す
- `docs/stablehlo-ops.md` の制御構造の節の「既知の制限」、`CLAUDE.md` の同じ項目の後半（「比較由来の `:i1` の carry を持つ while の結果を jit の戻り値にすると……」）、`docs/phase3-report.md` §5 の該当行

どのバグでも、撤去したら `docs/iree-upstream-bugs.md` の表と節に「何版で直ったか」を書き、`docs/iree-repros/` の該当ファイルと `scripts/check-iree-repros.sh` の `CASES` の行を消す。

## GPU で確かめる手順（issue #12）

同じ StableHLO から作った vmfb を `local`（CPU）と `cuda`（NVIDIA GPU）の
両方で実行し、結果が数値的に一致することを確かめる large テスト
（`tests/iree/cross-device-test.lisp`）がある。既定のテストスイート
（small + medium）には含まれず、`NABLA_TEST_SIZES=large` を明示したときだけ
実行される。GPU の無いこのマシンでは `skip-unless-cuda`
（`tests/iree/support.lisp`）が常にこのテストをスキップする。

### 手順（GPU があるマシンで）

```sh
# 1. CUDA HAL ドライバを含むランタイムをビルドする。コンパイラは既定の
#    PyPI ホイールに CUDA ターゲットバックエンドが含まれている
#    （契約 §0 事実3: GPU の無いこのマシンでも --iree-hal-target-device=cuda
#    でのコンパイルは成功する）。
scripts/build-iree.sh --cuda

# 2. large スイートを、CUDA が無ければ黙ってスキップせず失敗させる
#    NABLA_REQUIRE_CUDA=1 を立てて実行する。
NABLA_TEST_SIZES=large NABLA_REQUIRE_CUDA=1 scripts/run-tests.sh
```

### Colab の GPU で実行する

手元に NVIDIA GPU が無いときは、Colab CLI（google-colab-cli）で Colab の GPU VM を
借りて上の手順を実行できる。開発用のクラウド環境からは Colab に接続できない
（ネットワーク方針で拒否される）ので、手元のマシンで実行する。

```sh
uv tool install google-colab-cli      # 初回のみ。初回の colab 実行で OAuth の認証がある
scripts/colab-gpu-check.sh            # T4。--gpu L4 などで変える。--keep で VM を残す
```

`git archive HEAD` を VM に送り、`scripts/colab/remote-gpu-check.sh` が
`build-iree.sh --cuda` → `verify-iree.sh --cuda` → large テスト
（`NABLA_REQUIRE_CUDA=1`）→ 最大誤差の実測（`scripts/colab/measure-cross-device.lisp`）
を順に実行する。ログと要約（`summary.md`）は `colab-gpu-out/<セッション名>/` に
持ち帰る。下の結果表は `summary.md` の数字で埋める。

`tests/iree/cross-device-test.lisp` は add / matmul / reduce_sum のそれぞれ
f32 版・bf16 版（計6テスト）で、`local` と `cuda`（cuda-arch は指定せず
IREE の既定に任せる）に同じ乱数入力を渡し、結果を比較する
（`allclose` が不一致のとき最大誤差を表示する）。add は dtype ごとの既定の
許容誤差（f32: rtol 1e-5 / atol 1e-6、bf16: rtol 1e-2 / atol 1e-3）で、
matmul と reduce_sum は rtol 0・atol = `accumulation-atol`（総和の誤差の
上界 2·n·u·Σ|項|、`tests/support/dtypes.lisp`）で比べる。あわせて、同じ
テキストに対する vmfb ディスクキャッシュ（issue #10）のファイルが `local`
と `cuda` で別々にできていること（キャッシュのキーにターゲットが入って
いること）も確かめる。

### 実測（Colab Tesla T4）

2026-10-10 に Colab の Tesla T4（sm_75、ドライバ 580.82.07、CUDA 13.0、
nvcc 13.0.88）で測った。IREE は `third_party/iree.lock` の commit
（`e4a3b0405d7d23554da26403658d0e8c3c5ecf25`）で、コンパイラは PyPI
ホイール、ランタイムは `scripts/build-iree.sh --cuda` でソースビルドした
もの。cuda-arch は IREE の既定に任せた。入力はテストと同じ生成器
（`make-random-array`、seed 0..99 の 100 通り、値はおよそ [-1, 1)）で、
表は `scripts/colab/measure-cross-device.lisp` で作った（このスクリプトは
別の PR #177 にあり、main にはまだ無い）。

| fixture | dtype | 最大絶対誤差 | 最大相対誤差 | 旧既定の許容誤差を超えた seed |
| --- | --- | --- | --- | --- |
| add | f32 | 0 | 0 | 0 |
| add_bf16 | bf16 | 0 | 0 | 0 |
| matmul | f32 | 1.192e-7 | 2.177e-5 | 0 |
| matmul_bf16 | bf16 | 7.813e-3 | 3.765e-1 | 1 |
| reduce_sum | f32 | 0 | 0 | 0 |
| reduce_sum_bf16 | bf16 | 1.563e-2 | 4.286e-1 | 29 |

（最後の列は、各 dtype の旧既定の許容誤差（rtol / atol）で比べたときに
不一致になった seed の数）

bf16 の差はバグではなく、期待どおりの数値誤差である。`local`（llvm-cpu）
は bf16 の reduce や dot を f32 で累積して最後に1回だけ丸める（同じ入力で
`local` を実行した 400 行すべてがこのモデルと一致した）。一方、bf16 の
演算器を持たない sm_75 の `cuda` は bf16 の加算ごとに丸める。このため差は
出力の 1 ULP ではなく、途中で最大になった部分和の約 1 ULP になり、和が
打ち消し合うところでは出力の 192 ULP にも達する（相対誤差は 0.4 を
超える）。この2つの丸め方を真似たシミュレーションは表の値を桁まで
再現する。rtol はここでは意味を持たないので、テストは rtol 0・atol =
2·n·u·Σ|x|（Higham, *Accuracy and Stability of Numerical Algorithms*
§4.2）で比べる。実測の最大誤差はこの上界の約 1/32（reduce_sum_bf16）と
約 1/9（matmul_bf16）。

これは1種類の GPU とドライバでの測定で、他の GPU・ドライバ・IREE の版で
同じ結果になることは保証しない。新しい許容誤差で large スイートが GPU 上で
通ることはまだ確かめていない。issue #12 は、それを確かめるまで open の
ままにする。

## Lisp からの呼び出し（issue #6）

`nabla/iree` の実行時バインディング（`src/iree/runtime*.lisp`）は
`iree/runtime/*.h` の高水準 API を CFFI で直接呼ぶ。ビルドしたライブラリ固有の
注意点:

- **アロケータ**: `iree_allocator_system()` は `base/allocator.h` の
  `static inline` で、共有ライブラリからは export されていない。このビルドは
  `IREE_ALLOCATOR_SYSTEM=libc` で構成されているため、system allocator は
  `{self = NULL, ctl = iree_allocator_libc_ctl}` として組み立てる
  （`iree_allocator_libc_ctl` は `libnabla_iree_runtime.so` が export している）
- **構造体の値渡し・値返し**: `iree_allocator_t` / `iree_string_view_t` /
  `iree_const_byte_span_t` / `iree_hal_buffer_params_t` / `iree_timeout_t` は
  値で渡し、`iree_vm_module_signature` や `iree_vm_function_name` は構造体を
  値で返す。素の CFFI（SBCL の FFI）はどちらにも対応しないため、
  `cffi-libffi`（apt の `cl-cffi` に同梱。要 libffi-dev）をロードしてから
  `defcfun` / `defcstruct` を書くだけでよい。`cffi-libffi` が
  `cffi:*foreign-structures-by-value*` を差し替えるので、C 側のヘルパーを
  自分で書く必要はない
- **文字列ビュー**: `cffi:with-foreign-string` が返す長さは終端 NUL を含む。
  `iree_string_view_t` の `size` に渡すときは 1 引く（引かないとドライバ名の
  末尾に余計な 1 バイトが付き、`try_create_default_device` が本来存在する
  ドライバでも NOT_FOUND になる）
- **vmfb の識別子**: `ireeCompilerInvocationOutputVMBytecode` が出す vmfb は
  既定で「polyglot zip」形式（`--iree-vm-emit-polyglot-zip`）なので、先頭4バイトは
  ZIP local-file-header シグネチャ `50 4B 03 04`（"PK\3\4"）で、フラットバッファ
  自体の識別子ではない
- **invoke の入力検査**（issue #8）: コンパイルされた vmfb は、関数の入出力に
  `hal.buffer_view.assert` を自動で挿入する。そのため `nabla.iree:invoke` に
  StableHLO の宣言と違う形状・dtype・個数の buffer view を渡すと、プロセスが
  落ちたり未定義動作になったりせず、`iree_runtime_call_invoke` が
  `IREE_STATUS_INVALID_ARGUMENT` の `iree_status_t` を返す
  （`nabla.iree:iree-status-error` の `code` が `:invalid-argument` になる）
- **allocator の統計**（issue #11）: このビルドは `IREE_STATISTICS_ENABLE` が
  既定の1のまま（`scripts/build-iree.sh` はこれを切り替えない）なので、
  `iree_hal_allocator_query_statistics` は `local-task` で常に有効で、
  `device_bytes_allocated` / `device_bytes_freed` が実測値どおりに動く
  （2x3 の bf16 buffer view 1個で12バイトぶん増減することを確認済み）。
  `nabla.iree:device-allocator-statistics` から読める
- **finalizer と full GC の相性**（issue #11、根本原因は issue #5 で特定・修正
  済み）: この環境では、`with-device` / `with-session` を数百回作っては壊す
  既存のテスト（`RUNTIME/MAKE-DEVICE` など）や `compile-stablehlo` の呼び出し
  が積み重なった後に `(sb-ext:gc :full t)` を呼ぶと、SBCL が
  `garbage_collect: no SP known for thread` という fatal error で
  プロセスごと落ちることがあった（確率的で、毎回起きるわけではない）。
  これは `nabla.iree` の finalizer 機構そのもののバグではなく、
  `libIREECompiler.so` 内の LLVM が初回呼び出し中にプロセス全体のシグナル
  ハンドラを sigaction で登録し直し、SBCL が GC の stop-the-world に使う
  SIGUSR2 を上書きすることが根本原因だった（詳しい仕組みは
  `src/iree/signals.lisp` 冒頭のコメント）。`src/iree/signals.lisp` の
  `%register-llvm-signal-handlers`（LLVM の登録を、他の全 Lisp スレッドを
  止めた制御された1点で済ませる。`%call-with-world-stopped` は
  `src/ffi-support/signals.lisp`）と `with-lisp-signal-handlers-preserved`（`nabla/ffi-support`）
  （IREE を呼ぶ公開関数の本体を包み、ハンドラを元に戻す）で修正済み。
  `nabla.asd` で `finalizer-test` を device/session を大量に作るテストより
  前に置いているのは、この修正より前に発生頻度を下げるために採った緩和策の
  名残で、修正後はもう必須ではない（残るリスクは `signals.lisp` 冒頭に列挙
  してある）
