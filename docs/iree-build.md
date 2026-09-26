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
```

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
- `iree-compile`, `iree_compiler_API_SharedImpl` — `--compiler=source` のときだけ追加で
  ビルドする。後者は `compiler/src/iree/compiler/API/CMakeLists.txt` で
  `OUTPUT_NAME "IREECompiler"`, `SOVERSION 0` として定義され、`lib/libIREECompiler.so(.0)`
  を生成する

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
  libIREECompiler.so -> ../lib/libIREECompiler.so  # wheel モードのみ。iree-compile の
                                               # RUNPATH=$ORIGIN が要求する
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
  (4 コア / 15 GB RAM) では実用的な時間で終わらない。CI やより強力なマシンで再挑戦 |
| `scripts/build-iree.sh --cuda` | 未実施（環境に GPU/CUDA なし） | - | CUDA toolkit の
  あるマシンが必要 |
| `scripts/verify-iree.sh --cuda` | 未実施（環境に GPU/CUDA なし） | - | 同上 |

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
