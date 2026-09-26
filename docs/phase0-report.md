# フェーズ0 報告書: IREE 疎通

フェーズ0（issue #2、「Lisp から IREE を動かす」）で得た知見をまとめ、フェーズ1（#35）への引き継ぎ材料にする。

## 1. 範囲と結論

issue #2 の完了条件（5項目）に対する達成状況:

| 完了条件 | 判定 | 備考 |
| --- | --- | --- |
| 手書きの StableHLO が NVIDIA GPU（`cuda`）と CPU（`local`）の両方で動き、数値が一致する | 未（環境に GPU なし、#12 open） | `local` 単体は確認済み。`cuda` は開発環境に GPU が無く未検証 |
| 手元の環境（OS / CPU アーキ / CUDA 版）で IREE のリリース配布物が動くか、自前ビルドが要るかが記録されている | 達成 | 本報告書 §2、docs/iree-build.md |
| `backend` プロトコルの下に IREE 実装があり、vmfb のディスクキャッシュが効いている | 達成 | issue #9 / #10。§3(a) の実測で確認 |
| デバイスバッファがループ実行でリークしない | 達成 | issue #11。finalizer + `run-pending-finalizers` で確認（`tests/iree/finalizer-test.lisp`） |
| 既定のテストスイート（small + medium）が CI で通る | 達成 | `.github/workflows/ci.yml`（issue #13）。ローカルでも `scripts/run-tests.sh` で確認済み（本報告書 §3(b)） |

結論: **GPU での数値一致（#12）だけが環境の制約で未測定**で、それ以外の完了条件はすべて達成している。#12 は open のまま残す。フェーズ1（#35）はこの上に着手してよい。

## 2. 環境と IREE

- IREE v3.11.0 @ `e4a3b0405d7d23554da26403658d0e8c3c5ecf25`（`third_party/iree.lock` に記録）
- コンパイラ（`libIREECompiler.so`）は同じコミットからビルドされた PyPI ホイール `iree-base-compiler` 3.11.0（sha256 `ac3505591b6b134784eae7bcdf806fc66a2d120a82134b98bd4fbe488fdf84c5`、`third_party/iree.lock` に記録）を使う。ランタイム（`libnabla_iree_runtime.so`、`iree-compile` / `iree-run-module` / `iree-lld`）は常にソースからビルドする（約10秒）
- フルソースビルド（`scripts/build-iree.sh --compiler=source`）は、この環境（4コア）では 59分かけて ninja 5674/7085 ステップで打ち切った。CI（GitHub Actions のランナー）や手元の非力なマシンでは実用的な時間で終わらない
- `iree-lld` を `--iree-llvmcpu-embedded-linker-path` で明示的に渡している（後述 §4）
- SBCL 2.2.9（apt パッケージ）。Quicklisp は使わない。apt の Common Lisp ライブラリ（`cl-fiveam` など）に加え、check-it / optima を固定コミットで git clone している
- `nabla/iree` は `cffi-libffi`（apt の `libffi-dev` が要る）に依存する
- GPU / CUDA なし（`nvidia-smi` が無い環境）。`--cuda` を付けた `cmake` の configure は NVIDIA の redistributable package index への 403 で失敗し、CUDA ターゲットは未検証
- IREE コンパイラの PyPI ホイールモード（既定）は x86_64 Linux 専用。他のプラットフォームでは `--compiler=source` を使う

## 3. 実測

### (a) `backend-compile` のコンパイル時間（キャッシュミス / ヒット）

`nabla:*compile-cache-directory*` を一時ディレクトリに束縛し（既存キャッシュの影響を除くため）、6つのフィクスチャ（add / matmul / reduce_sum × f32 / bf16）について `backend-compile` を2回呼び、`get-internal-real-time` で計測した。測定スクリプトは scratchpad に置いた一回限りのもので、リポジトリには入れていない。

| フィクスチャ | 初回（キャッシュミス） | 2回目（キャッシュヒット） |
| --- | --- | --- |
| add | 384.0 ms | 0.0 ms |
| add_bf16 | 356.0 ms | 0.0 ms |
| matmul | 344.0 ms | 4.0 ms |
| matmul_bf16 | 356.0 ms | 0.0 ms |
| reduce_sum | 348.0 ms | 0.0 ms |
| reduce_sum_bf16 | 352.0 ms | 0.0 ms |

単一 op のコンパイルは数百 ms、キャッシュヒットは 1 ms 未満（誤差の範囲）という、計画タブの見込みどおりの結果になった。README の使用例（add、shape 4）でも同様の傾向（初回 約380〜400 ms、2回目 0 ms）を確認している。

### (b) `scripts/run-tests.sh` の wall time

- 既定（small + medium）: 607〜608 checks、すべて pass、実測 約55秒（`real 0m54.801s`〜`0m55.335s`、実行のたびに数百ms〜数秒のばらつきあり）
- `NABLA_TEST_SIZES=large scripts/run-tests.sh`: GPU が無いため cuda 系のテストはすべて `fiveam:skip`（失敗ではない）。実測 約1分16秒（`real 1m15.581s`）

### (c) CI（GitHub Actions、issue #13）

CLAUDE.md に記録済みの実測（PR #22、ubuntu-24.04）: IREE キャッシュが無いとき（cold）はジョブ全体で約4分43秒（うち `scripts/build-iree.sh` が約3分4秒）、キャッシュが当たったとき（warm）は約44秒。環境によって変わる目安値。

### (d) `scripts/build-iree.sh`

べき等な再実行（このマシンでは既にビルド済みの状態からの再実行）は、実行のたびに 4秒（`[build-iree] elapsed: 4s`、`real 0m3.608s`）〜1分40秒（`real 1m40.035s`）とばらつきがあった。CLAUDE.md の「数秒」は、キャッシュされたビルドディレクトリを ninja が no-op で確認する場合の値で、リンクや検証まで含めて計測すると本報告書の実測レンジ（4秒〜1分40秒程度）になる。

### (e) `local` vs `cuda`

- コンパイル: 両方とも成功する。`tests/iree/backend-test.lisp` の `BACKEND/CUDA-TARGET/COMPILES-ALL-FIXTURES-WITHOUT-A-GPU` が、GPU の無い環境でも `:cuda` ターゲット向けのコンパイルが通ることを確認している（実行はしない）
- 実行結果の比較: 未測定（GPU が無いため。`tests/iree/cross-device-test.lisp` の該当テストはすべてスキップされる）

## 4. C API で計画と違った点

| 項目 | 計画の記述 | 実際 | 対処したファイル |
| --- | --- | --- | --- |
| コンパイラのロード | `libIREECompiler` を `ireeCompilerLoadLibrary` でロードする（設計タブ） | `ireeCompilerLoadLibrary` は `libIREECompiler.so` から export される関数ではない。dlopen 側（呼び出す側）が自分で用意するローダ関数の名前だった | `src/iree/library.lisp`（`cffi:load-foreign-library` で代替） |
| vmfb の形式 | 特に記述なし | vmfb は「polyglot zip」形式で出力される。先頭4バイトは ZIP の local-file-header シグネチャ `PK\3\4`（`#x50 #x4B #x03 #x04`）で、フラットバッファ自体の識別子ではない | `tests/iree/support.lisp` の `*vmfb-magic*`、`docs/glossary.md` の vmfb 項 |
| 構造体の受け渡し | 特に記述なし | `iree_allocator_t` / `iree_string_view_t` / `iree_hal_buffer_params_t` / `iree_timeout_t` など、値渡し・値返しの構造体が多く、素の CFFI では扱えない | `nabla/iree` 全体が `cffi-libffi` に依存（`libffi-dev` が要る） |
| `iree_allocator_system` | 特に記述なし | ヘッダ上は `static inline` 関数で、共有ライブラリからは呼べない。`{NULL, iree_allocator_libc_ctl}` の構造体を Lisp 側で組み立てて代用する | `src/iree/runtime-ffi.lisp` |
| LLVM のシグナルハンドラ | 特に記述なし | LLVM は初回呼び出し中にプロセス全体のシグナルハンドラを sigaction で登録し直し、SBCL が GC の stop-the-world に使う SIGUSR2 を上書きする。放置すると、以後どこかのスレッドが GC を始めた瞬間に "no SP known for thread" で SBCL が確実に落ちる（issue #5） | `src/iree/signals.lisp`（`ensure-compiler-loaded` が `%call-with-world-stopped` で他の全 Lisp スレッドを止めた、制御された1点で登録を済ませる。`with-lisp-signal-handlers-preserved` で多重に防御する）。残るリスクは本報告書 §5 の新規リスク (i) を参照 |
| コンパイラ呼び出しの方式 | 埋め込み C API を第一候補、`iree-compile` のサブプロセス起動を代替案とし、フェーズ0で両方試して決める | 埋め込み C API（`ireeCompilerSessionCreate` → `ireeCompilerSessionSetFlags` → 入力を渡して出力バッファに vmfb を受け取る）に決定。サブプロセス方式は採らない。ただし例外が1つある: `llvm-cpu` ターゲットは実行体のリンクに外部リンカ（`iree-lld`）をサブプロセス起動する（IREE 側にプロセス内リンクの手段が無いため）。「サブプロセスを起動しない」という方針の唯一の例外になる | `src/iree/compiler.lisp`、`scripts/build-iree.sh`（`--iree-llvmcpu-embedded-linker-path` で `iree-lld` を明示） |
| `iree_string_view_t` のサイズ | 特に記述なし | `size` は NUL 終端を含まない（C 文字列としての長さそのもの） | `src/iree/runtime-ffi.lisp` |
| 引数の形状・dtype 検証 | 特に記述なし | `hal.buffer_view.assert` により、宣言と違う形状・dtype の引数を渡すと `iree-status-error`（code `:invalid-argument`）が signal されるだけで、プロセスがクラッシュすることはない | `tests/iree/execute-test.lisp` |
| `backend` プロトコルの関数名 | 設計タブでは `compile`, `load`, `invoke`, `to-device`, `to-host` | CL の `compile` / `load` と衝突するため、総称関数はすべて `backend-` 接頭辞にした（`backend-compile`, `backend-load`, `backend-unload`, `backend-invoke`）。issue #9 の文言とも異なる | `src/backend.lisp` |
| SBCL の GC と LLVM のシグナルハンドラの相性 | 特に記述なし | full GC 中に fatal error（`garbage_collect: no SP known for thread`）が確率的に出ることがあった。根本原因は LLVM（`libIREECompiler.so` 内）が初回呼び出し中にプロセス全体のシグナルハンドラを sigaction で登録し直し、SBCL が GC の stop-the-world に使う SIGUSR2 を上書きすることだと特定し、`src/iree/signals.lisp` で修正済み（issue #5） | `src/iree/signals.lisp`（`%register-llvm-signal-handlers` / `with-lisp-signal-handlers-preserved`）、`nabla.asd` のコメント、本報告書 §5 の新規リスク (i)、§8 |

## 5. リスク表の再評価

計画タブ「リスクと対策」の各行を、フェーズ0の結果で再評価する。

| リスク | フェーズ0の結果 | 根拠 |
| --- | --- | --- |
| コンパイルが遅い（演算1つでも数十〜数百ms、ResNet 級の学習ステップは lean4-mlir の実測で 10〜15 分） | 確認済み | §3(a) の実測（単一 op で約350ms）。対策（vmfb キャッシュ + eager 実装）は #10 で実装済み |
| StableHLO方言のバージョン変化 | 残存 | フェーズ0では固定コミットに追従するだけで、バージョン変化そのものはまだ経験していない。CI で `third_party/iree.lock` をキーにキャッシュしており、追従自体の仕組みはある |
| IREE未対応のStableHLO op（一部rng、custom_call等） | 残存 | フェーズ0で使った op（add / matmul / reduce_sum、f32・bf16）はすべて対応していた。op 対応表は未作成（フェーズ1の #30 で作る） |
| 非標準な GPU 環境（例: DGX Spark の aarch64 + CUDA 13 + sm_121） | 該当せず（未確認） | 開発環境に GPU が無く未確認（#12 open）。x86_64 Linux では wheel + ソースビルドしたランタイムで CPU 動作を確認した |
| ランタイム API 名が版で変わる | 確認済み | フェーズ0で3点発見: `ireeCompilerLoadLibrary` は `libIREECompiler.so` から export されない（dlopen 側のローダ）。`iree_allocator_system` は `static inline`。構造体の値渡し・値返しには `cffi-libffi` が要る（§4） |
| 配布: C ランタイムの公開パッケージがない | 確認済み（対策を更新） | ランタイムは固定コミットからソースビルド（約10秒）。コンパイラは同じコミットの PyPI ホイールを sha256 固定で使う（フルソースビルドは4コアで59分・7085ステップ中5674で打ち切り）。詳細は §2、§6 |
| IREE 経路で ImageNet 規模の学習が届くか未確認（lean4-mlir でも開いた問題） | 該当せず | フェーズ0のスコープ外（手書き StableHLO の疎通のみ）。フェーズ1以降で評価する |
| デバイスメモリとLisp GCの不整合 | 確認済み（実装済み） | #11 で実装済み。`device-array` は生成時に device を retain し、解放時に buffer view → device の順で release する。finalizer は別スレッドで非同期に走るため、テストでは `gc-and-run-finalizers`（`run-pending-finalizers` を明示的に呼ぶ）を使う |
| 動的形状 | 該当せず（未着手） | フェーズ0は静的形状のみを扱う手書きフィクスチャで、動的形状は経験していない |
| ADのルール数が多い | 該当せず | フェーズ0のスコープ外（AD はフェーズ2） |
| CFFI越えのオーバーヘッド | 該当せず（未計測） | フェーズ0では単発呼び出しの実測（§3(a)）はあるが、`jit` した関数を繰り返し呼ぶケースでのオーバーヘッド比較はまだしていない |

新規リスク（フェーズ0で発見したもの。計画タブに追加を提案する）:

| リスク | 影響 | 対策 |
| --- | --- | --- |
| （i）LLVM のシグナルハンドラ上書き（解決済み） | LLVM を含むコンパイラの初回呼び出し後、GC の stop-the-world（SIGUSR2）が壊れて SBCL が "no SP known for thread" で落ちる。§4 の「SBCL の GC と LLVM のシグナルハンドラの相性」行、および §8 で「テストの実行順序による緩和のみで根本原因は未調査」としていたフレーキーな fatal error は、この同じ原因だったと判明した | `ensure-compiler-loaded` が全 Lisp スレッドを止めた1点でシグナルハンドラの登録を済ませ、`with-lisp-signal-handlers-preserved` で呼び出しごとに多重防御する（`src/iree/signals.lisp`）。`nabla.asd` の `finalizer-test` を先に置くテスト順序は、この修正より前の緩和策の名残で、修正後はもう必須ではない。残るリスク（世界を止めている間の他ロック待ち・シグナル配送・GC ロックの餓死）は `signals.lisp` 冒頭のコメントに列挙してある |
| （ii）wheel 配布の x86_64 Linux 限定、`--compiler=source` 未検証 | 他プラットフォーム（macOS、aarch64 Linux 等）では既定のセットアップが使えない可能性がある | `scripts/build-iree.sh --compiler=source` を用意してあるが、実際に他プラットフォームで通したことはない（CI でも未検証） |

## 6. 計画 Artifact への提案編集

計画 Artifact 自体は編集していない（編集禁止）。次の編集を提案する。旧テキストは Artifact を実際に読んで確認した現行の文言をそのまま引用している。

| タブ | 節 | 旧テキスト | 新テキスト |
| --- | --- | --- | --- |
| 計画 | フェーズ別ロードマップ（表の直後の段落） | 「フェーズ1と2の間で、eagerモードの扱いを決める（後述のリスク参照）。フェーズ4以降はユーザーを増やせる段階なので、READMEとチュートリアルもこの時点で用意する。」 | 同段落の先頭に追記: 「フェーズ0は 2026-09-26 に完了（CPU）。GPU での数値一致（#12）のみ未測定で open。」 |
| 計画 | リスクと対策（「配布: C ランタイムの公開パッケージがない」行の対策セル） | 「ソースからビルドする（決定済み、Python 不使用）。ビルド手順と固定コミットをリポジトリに同梱する」 | 「ランタイムは固定コミットからソースビルド（約10秒）。コンパイラ (libIREECompiler.so) は同じコミットの PyPI ホイールを sha256 固定で使う（フルソースビルドは 4 コアで 59 分・7085 ステップ中 5674 で打ち切り）。Python はホイールの取得にだけ使い実行時依存にしない。手順は scripts/build-iree.sh と docs/iree-build.md」 |
| 計画 | リスクと対策（「ランタイム API 名が版で変わる」行の対策セル） | 「固定した IREE バージョンのヘッダから写す。設計書の API 名は目安」 | 末尾に追記: 「フェーズ0で確認: ireeCompilerLoadLibrary は libIREECompiler.so から export されない（dlopen 側が用意するローダ）。iree_allocator_system は static inline。構造体の値渡し・値返しには cffi-libffi が要る」 |
| 計画 | リスクと対策（「非標準な GPU 環境」行の対策セル） | 「フェーズ0で手元の環境での可否を確認する。動かなければ自前ビルド。IREE の CUDA ターゲットは `--iree-cuda-target=sm_XX` で世代を指定するので、LLVM 側が対応する世代かも確認」 | 末尾に追記: 「フェーズ0の開発環境には GPU が無く未確認（#12 open）。x86_64 Linux では wheel + ソースランタイムで CPU 動作を確認した」 |
| 計画 | リスクと対策（「デバイスメモリとLisp GCの不整合」行の対策セル） | 「バッファをCLOSオブジェクトで包み、trivial-garbage の finalizer で解放。明示的 `free` も提供」 | 末尾に追記: 「実装済み（#11）。device-array は device を retain し、buffer view → device の順で release する。SBCL の finalizer は別スレッドで走るため、テストでは run-pending-finalizers を使う」 |
| 計画 | リスクと対策（「コンパイルが遅い」行の対策セル） | 「演算単位の vmfb キャッシュに加え、eager専用の自前CPU実装を持つ（決定済み）。全プリミティブの第二実装であり、工数はフェーズ1〜2級と見込む。フェーズ1以降、各プリミティブは StableHLO 出力と CPU 実装を同時に書く。本番は `jit` 前提とする」 | 末尾に追記: 「フェーズ0の実測（単一 op のコンパイル約350ms、vmfb キャッシュヒット時は1ms未満）」 |
| 計画 | リスクと対策（表の末尾に新規2行を追加） | （無し） | §5 の新規リスク (i)(ii) の2行をそのまま追加 |
| 計画 | 最初の2週間でやること（チェックリスト） | 項目1〜9（本文参照） | 項目2〜6, 8, 9 にチェックを入れる。項目1（IREE を固定コミットでソースからビルドし... `cuda` ターゲットでコンパイル・実行できることを確認する）と項目7（同じ StableHLO を `local` と `cuda` 向けにコンパイルし、結果の数値一致を確認する）は「CPU は確認済み、CUDA は GPU 環境待ち（#12）」を付記して未完のままにする |
| 設計 | StableHLO出力と実行系連携の設計（コンパイラ呼び出しの段落） | 「（libIREECompiler を ireeCompilerLoadLibrary でロードし、セッションにフラグを与えて入力を渡す方式）を第一候補とし、iree-compile のサブプロセス起動を代替案とする。埋め込み API は共有ライブラリを dlopen して使う設計なので CFFI と相性が良く、テキストをファイル経由で渡す必要もない。どちらを採るかはフェーズ0で両方試して決める」 | 「（libIREECompiler.so を CFFI で dlopen し、ireeCompilerGlobalInitialize → セッション → フラグ → メモリ上の入力 → 出力バッファ）に決定。ireeCompilerLoadLibrary はライブラリ側が export する関数ではないので使わない。例外として llvm-cpu は実行体のリンクに iree-lld をサブプロセス起動する（IREE 側にプロセス内リンクの手段が無い）」 |
| 設計 | StableHLO出力と実行系連携の設計（テキスト出力の方針の箇条書き「IREE はソースからビルドして...」） | 「IREE はソースからビルドして libIREECompiler.so とランタイム共有ライブラリを得る（lean4-mlir と同じ方針）。Python は使わない。ビルド手順はリポジトリに同梱する」 | 上の「配布」行の新テキストと同じ内容に差し替える |
| 設計 | StableHLO出力と実行系連携の設計（「vmfb はテキストの SHA-256 と...」の箇条書き） | 「vmfb はテキストの SHA-256 とコンパイルターゲット（デバイス種別＋アーキ）をキーにディスク（~/.cache/nabla/）に保存し、2回目以降はコンパイルしない。lean4-mlir の実測ではモデル級のコンパイルに分単位かかるため、これは必須機能」 | 末尾に追記: 「実装: キーは backend-fingerprint（IREE リビジョン・target・cuda-arch・解決済みフラグ・local では CPU model name）と TEXT の長さ接頭辞つき SHA-256。ファイルは NBLMOD01 + payload の SHA-256 + vmfb、tmp + rename で原子的に書く（src/compile-cache.lisp）」 |
| 設計 | StableHLO出力と実行系連携の設計（「MLIR 診断は Lisp のコンディションに変換し...」の箇条書き） | 「MLIR 診断は Lisp のコンディションに変換し、どの eqn が失敗したかを報告する（テキスト内に loc("eqn-42") を付けておく）」 | 末尾に追記: 「フェーズ0: iree-compile-error（phase / diagnostics / message）まで実装。loc の付与はフェーズ1の emitter の課題」 |
| 設計 | StableHLO出力と実行系連携の設計（ランタイム対応表） | `iree_hal_device` 行「device オブジェクト（(device :cuda 0) 等）」／`iree_runtime_session + vmfb` 行「compiled-function」／`iree_hal_buffer_view` 行「GC 管理 + 明示的解放」 | `iree_hal_device` 行を「(make-device :local \| :cuda)（nabla.iree）」に、`iree_runtime_session + vmfb` 行を「iree-module（backend-load の返り値、backend-unload で解放）」に差し替え、`iree_hal_buffer_view` 行に「device を retain」を追記。表の前の注記「以下の関数名は目安」はそのまま |
| 設計 | バックエンドの選択（backend プロトコルの段落） | 「バックエンド層は backend プロトコル（compile, load, invoke, to-device, to-host）として抽象化し、...」 | 「バックエンド層は backend プロトコル（make-backend / find-backend, backend-compile, backend-load, backend-unload, backend-invoke, to-device, to-host, device-array-aval, backend-target, backend-fingerprint。CL の compile / load と衝突するため backend- 接頭辞）として抽象化し、...」 |
| 設計 | 全体アーキテクチャ（パッケージ構成表の `nabla` 行） | 「コア: トレーサ、IR、変換（jit / grad / vmap）、StableHLO出力、device-array、backend プロトコル」 | 「コア: トレーサ、IR、変換（jit / grad / vmap）、StableHLO出力、backend プロトコルと device-array の総称関数（to-device / to-host / device-array-aval）。device-array クラス自体は実行系側（nabla.iree）」 |
| 設計 | デバイスバッファとメモリ管理（dtype の段落） | 「dtype は single-float, double-float, (signed-byte ...), (unsigned-byte ...), bit に加え、bf16 と f16...」 | 末尾に追記: 「フェーズ0の dtype タグは :f32 :f64 :bf16 :f16 の4つ（src/dtype.lisp）。整数型はフェーズ1以降」 |
| 設計 | 全体アーキテクチャ（末尾の段落） | 「grad / vmap はIRを別のIRに書き換える変換で、jit だけがIRをStableHLOに落として外へ出す。実行系は backend プロトコルの背後に隠し、PJRT ならクライアント（プラットフォーム）を、IREE なら HAL ドライバを切り替えることでデバイスを選ぶ。コンパイル済みの実行体はターゲットごとに1つ持つ。」 | 末尾に追記: 「nabla/iree は共有ライブラリが無くてもロードでき、make-backend :iree が iree-library-not-found を signal する（実装済み）」 |

## 7. フェーズ1への引き継ぎ

- 作成した issue: 親 #35（フェーズ1: トレースと jit）、子 #29（IR と defprimitive の骨格）、#30（op 対応表）、#31（プリミティブ集合）、#32（トレーサ）、#33（emitter）、#34（jit とキャッシュ）
- フェーズ1が前提にできる公開 API は README.md の「公開 API」節（dtype、aval、backend プロトコル、`*compile-cache-directory*`）と `nabla.iree` の `iree-backend` / `compile-stablehlo` / `compile-flags` / コンディション階層に固定してある。これ以上は増やしていない
- `backend-invoke` は `"module.<name>"`（固定のモジュール名 `module` の後に関数名を付けたもの）を仮定している（`src/iree/backend.lisp`）。そのため、フェーズ1の emitter（#33）は名前付きモジュールではなく、無名の `builtin.module` の中に `func.func @main` を出すこと
- `compile-flags` の docstring は「iree-lld の有無という隠れた入力」がある旨を警告している。フェーズ1の `jit`（#34）のキャッシュキーは `backend-fingerprint` を経由してこれを含むので、直接気にする必要はないが、`jit` 独自のキャッシュ層を作るときは同じ注意が要る
- テストで使うヘルパー: `skip-unless-iree`（IREE の共有ライブラリが無ければスキップ）、`define-iree-test` / `define-iree-test/large`（`:nabla.medium` / `:nabla.large` に登録する fiveam:test の代わり）、`stablehlo-fixture`（`tests/fixtures/stablehlo/` の内容を読む）。この3つは `tests/iree/support.lisp` にある。`regression-path :package "..."`（`tests/regressions/` にファイルを置くときのパスの慣習）は `nabla/test-support`（`tests/support/regression.lisp`）にある

## 8. 未解決・次のアクション

- **#12 の GPU 測定**: GPU が用意でき次第、`docs/iree-build.md` の GPU 実測表を埋め、`NABLA_TEST_SIZES=large NABLA_REQUIRE_CUDA=1 scripts/run-tests.sh` を実行して local/cuda 数値一致を確認する
- **full GC × IREE スレッドの fatal error（解決済み）**: `garbage_collect: no SP known for thread` で稀にプロセスが落ちる fatal error（issue #5）は、LLVM（`libIREECompiler.so` 内）が初回呼び出し中にプロセス全体のシグナルハンドラを sigaction で登録し直し、SBCL が GC の stop-the-world に使う SIGUSR2 を上書きすることが根本原因だと特定した。`src/iree/signals.lisp` の `%register-llvm-signal-handlers`（LLVM の登録を、他の全 Lisp スレッドを止めた制御された1点で `ireeCompilerOutputOpenMembuffer` / `ireeCompilerOutputDestroy` により済ませる）と `with-lisp-signal-handlers-preserved`（IREE を呼ぶ公開関数の本体を包み、ハンドラを元に戻す多重防御）で修正済み。`nabla.asd` の `finalizer-test` を先に置くテスト実行順序は、この修正より前に発生頻度を下げていた緩和策の名残で、もう必須ではない。残るリスク（世界を止めている間の他ロック待ち・GC 以外のシグナル配送・GC ロックの餓死）は解消しておらず、`src/iree/signals.lisp` 冒頭のコメントに列挙してある
- **`.gitignore` の矛盾**: 「`tests/*.lisp` と `.claude/settings.json` が ignore 対象なのに追跡されている」という矛盾が過去に指摘されていたが、本ユニットの作業時点で `git check-ignore` で確認した限り再現しなかった（この2つのパスは現在の `.gitignore` の内容とは一致していない）。もし何らかの理由で再発したら、別の `chore:` PR で直す
- **`--compiler=source` の CI での検証**: 現状 CI は `--compiler=wheel`（既定）のみを使っている。フルソースビルドは時間の制約で未検証（本報告書 §2）
