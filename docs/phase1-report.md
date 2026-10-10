# フェーズ1 報告書: トレースと jit

フェーズ1（親 issue #35、「Lisp 関数から StableHLO を出す」）で作ったもの・得た知見をまとめ、フェーズ2（grad）への引き継ぎ材料にする。

## 1. 範囲と結論

#35 の完了条件（5項目）に対する達成状況:

| 完了条件 | 判定 | 検証方法 |
| --- | --- | --- |
| 要素演算・matmul・reduce・reshape・broadcast を含む Lisp 関数が `jit` で `local` 上で動き、eager 実装および JAX フィクスチャと数値一致する | 達成 | `tests/iree/jit-test.lisp` の `jit/mlp-matches-jax-fixture-and-eval-graph`（線形層2つ + tanh の小さな MLP を `defjit` し、`local` backend の結果が JAX フィクスチャと `eval-graph` の両方に許容誤差内で一致することを確認）。プリミティブ単体は各 `tests/iree/*-test.lisp` の medium PBT（IREE の結果と eager の一致） |
| 同じキー（関数・aval・静的引数・ターゲット）では再コンパイルしない | 達成 | `tests/iree/jit-test.lisp` の `jit/does-not-recompile-on-repeated-call`（`nb::*jit-miss-count*` が2回目の呼び出しで増えないことを確認）。ディスクキャッシュ側は `tests/iree/compile-cache-test.lisp` |
| op 対応表がある | 達成 | `docs/stablehlo-ops.md`（issue #30）。全行に `tests/fixtures/stablehlo/ops/<name>.mlir` フィクスチャがあり、`tests/iree/ops-test.lisp` が実際にコンパイルできることを確かめる |
| 既定のテストスイート（small + medium）が CI で通る | 達成 | `.github/workflows/ci.yml`。ローカルでも本報告書 §4 で確認した |
| 各プリミティブに形状推論・StableHLO 出力・eager 実装の3つが揃っている | 達成 | `tests/primitives/registry-test.lisp` の `registry/all-phase1-primitives-have-abstract-eval-emit-and-eager`（フェーズ1の19プリミティブ全部について `find-primitive` が `:abstract-eval`・`:emit`・`:eager` の3つとも non-NIL であることを検査） |

結論: #35 の完了条件はすべて達成している。ただし §5 で述べる通り、各子 issue の個別の完了条件にあった「mutation score 80%以上」は、`tools/mutate` の演算子の少なさ・対象範囲の狭さのため弱くしか検証できていない。

## 2. 作ったもの

### 2.1 IR と defprimitive（#29、PR #41 / #44 / #48）

`aval`（形状＋dtype）/ `var` / `eqn` / `graph` の4つの構造体と、`defprimitive` マクロを `src/primitive.lisp` / `src/ir.lisp` に実装した。`defprimitive` は1つの演算に `:abstract-eval`（形状推論）・`:emit`（StableHLO 出力）・`:eager`（CPU 実装）の3つを束ねる（CLAUDE.md の設計上の約束どおり、jvp / transpose ルール・バッチ化ルールは未実装で、これがフェーズ2/3の仕事になる）。

`src/ir-print.lisp`（PR #44 / #48）に `print-graph`（export）と内部の `read-graph` を実装し、graph を jaxpr 風のテキストに変換・読み戻せるようにした。この往復（round-trip）の性質テストとレビューで、2つのバグが見つかった（§4.6）。

### 2.2 op 対応表（#30、PR #40）

`docs/stablehlo-ops.md` に、emitter が出す StableHLO op と IREE が受け付ける綴りの対応表を作った。全行がフィクスチャ（`tests/fixtures/stablehlo/ops/`）を持ち、`tests/iree/ops-test.lisp` がコンパイル可能性だけを検査する（数値の正しさはプリミティブ実装側の責任）。

### 2.3 フェーズ1のプリミティブ集合、19個（#31、PR #42・#43・#45・#46・#47・#49・#50・#51・#52）

`src/primitives/` 以下に以下の19個を実装した（`tests/primitives/registry-test.lisp` の `*phase1-primitive-names*` が正）:

- 二項算術（`arith.lisp`）: `add` `sub` `mul` `div`
- 単項（`unary.lisp`）: `neg` `exp` `log` `tanh`
- 比較・選択（`compare.lisp`）: `compare` `select` `convert`、`max` `min`
- 形状（`shape.lisp`）: `reshape` `broadcast-in-dim` `transpose`
- 縮約（`dot.lisp`）: `dot-general`
- reduce（`reduce.lisp`）: `reduce-sum` `reduce-max`

このほか issue #37（PR #42）で真偽値の dtype `:i1` を、issue #38（PR #43）で bf16 / f16 と single-float の最近接偶数丸め変換（`src/float16.lisp`）を追加した。issue #39（PR #47）で `eval-graph`（graph を各プリミティブの `:eager` 実装だけで CPU 上評価するインタプリタ。`src/eval.lisp`）を実装し、JAX フィクスチャに頼らず自前 IR の意味論を検査できるようにした。

### 2.4 トレーサ（#32、PR #57 / #61）

`src/walk.lisp` の `%walk` によるコードウォークで `with-tracing` マクロを実装した（`src/trace.lisp` / `src/trace-ops.lisp`）。CL の標準関数呼び出しを内部演算に書き換え、`setq` など未対応の形式は `unsupported-form` にする。`if`（および `when`/`unless`/`cond`/`and`/`or` のように展開されるもの）は、条件が `:i1` のトレーサ・配列なら THEN・ELSE を両方評価してから `select` に書き換える（PR #61）。配列レベルの公開 API（`dot` `reshape` `transpose` `broadcast-in-dim` `reduce-sum` `reduce-max` `convert` `where`、`src/array-api.lisp`）もここで揃えた。

### 2.5 StableHLO テキスト emitter（#33、PR #56）

`src/stablehlo.lisp` の `emit-stablehlo` が graph を、無名の module 中に1つの `func.func @main` を持つ StableHLO テキストへ変換する。各 eqn の出力行に `loc("eqn-N")` を付け、コンパイルエラー時にどの eqn が原因かを逆引きできるようにした（フェーズ0で「フェーズ1の課題」としていたもの）。

### 2.6 jit とインメモリキャッシュ（#34、PR #66 / #67）

`src/jit.lisp` に `jit`（`with-tracing` が作った `traceable-function` を、呼ぶたびに必要なら1回だけコンパイルする `jitted-function` にする）と `defjit`（トレース対象の関数をその場で定義する糖衣マクロ）を実装した。パイプラインは trace → graph → emit → compile → load → execute の6段階に分け、`%jit-cache-lookup-or-compile` の中で trace と emit の間に置く（フェーズ2/3 の書き換えが挟まる場所。§6 参照）。インメモリのキャッシュ（`*jit-cache*`）は関数の同一性（EQ）・aval・静的引数・backend（フィンガープリント込み）をキーにし、vmfb のディスクキャッシュ（`src/compile-cache.lisp`、フェーズ0）とは独立した層になっている。コンパイル失敗は `jit-compile-error`（失敗した graph と、分かれば原因の eqn を持つ）に変換され、`use-eager`（この呼び出しだけ `eval-graph` にフォールバック）・`recompile`（もう一度コンパイルする）の2つのリスタートを提供する。

## 3. 完了条件の検証（詳細）

- **数値一致**: `tests/iree/jit-test.lisp` の `jit/mlp-matches-jax-fixture-and-eval-graph` が、`defjit` した小さな MLP の `local` backend の出力を JAX フィクスチャと `eval-graph` の両方に対して許容誤差内で比較する。各プリミティブ単体は `tests/iree/arith-test.lisp` / `unary-test.lisp` / `compare-test.lisp` / `dot-test.lisp` / `reduce-test.lisp` / `shape-test.lisp` の medium PBT が、IREE 実行結果と `eval-graph` の一致を dtype ごとの許容誤差でランダム形状に対して検査する
- **再コンパイルしない**: `jit/does-not-recompile-on-repeated-call`（`nb::*jit-miss-count*` を検査）。ディスクキャッシュ側は `tests/iree/compile-cache-test.lisp`
- **op 対応表**: `docs/stablehlo-ops.md` + `tests/iree/ops-test.lisp`
- **既定スイートが CI で通る**: `.github/workflows/ci.yml`。本報告書 §4 でローカル実行も確認した
- **各プリミティブの3点セット**: `tests/primitives/registry-test.lisp` の `registry/all-phase1-primitives-have-abstract-eval-emit-and-eager`（19個全部を検査）と `registry/case-list-covers-exactly-the-phase1-names`（テスト自身の名前の書き漏らしがないことの確認）

## 4. 教訓と落とし穴

### 4.1 SBCL の浮動小数点トラップを IREE のワーカースレッドが引き継ぐ（#53、PR #58）

IREE（`local-task`）のワーカースレッドは、生成時に SBCL の「トラップを外していない」SSE/x87 の浮動小数点例外マスクをそのまま引き継ぐ。NaN を含む入力に順序つき比較（`stablehlo.compare`）や `stablehlo.maximum` を実行すると invalid フラグが立ってトラップが発火し、SBCL のプロセスごと落ちる（`stablehlo.add` は NaN をそのまま返すので落ちない）。デバイス・ワーカースレッドの作成とコンパイラの呼び出しは、浮動小数点トラップを外した状態で行う必要がある（`src/iree/signals.lisp` 周辺）。同じ系統の不具合として、縮約次元が0の `dot_general` をコンパイルすると Lisp の生の `DIVISION-BY-ZERO` が signal される問題も同じ issue で見つかった（コンパイラの FFI 経路にはトラップのマスクが無い）。

### 4.2 IREE の llvm-cpu が bf16 / f16 の dot と reduce を f32 で累積しない（#54/#59、#63/#64）

eager 実装は single-float で累積してから最後に1回だけ丸めるが、IREE（llvm-cpu）は bf16/f16 の `dot_general` / `reduce` を素の bf16/f16 のまま累積する。縮約の長さが大きいと（K=64 で bf16 は200回中103回、f16 は45回）許容誤差を超えてずれる。対策は `dot-general` / `reduce-sum` の `:emit` を、bf16/f16 のときは f32 の結果型で出してから `stablehlo.convert` で元の dtype に戻す複数行の出力にすること（`docs/stablehlo-ops.md` に記録）。今の medium テストが元々通っていたのは、テストの縮約長がたまたま小さかったから、という点が怖い教訓。

### 4.3 IREE の演算融合で丸めの回数が eager とずれる（#67 のレビュー知見）

IREE はコンパイル時に複数の演算を融合できるため、eager 実装が「演算ごとに1回丸める」のに対し、IREE 側は「融合したブロック全体で1回だけ丸める」ことがある。bf16/f16 のテストで eager と IREE を直接比較すると、この丸め回数の違いだけで許容誤差を超えてずれることがある。対策として、bf16/f16 の一致検証には f32 で計算した「オラクル（真の値に近い基準）」を経由し、eager・IREE の両方をそのオラクルと比較する（互いを直接比較しない）方式にした。

### 4.4 IREE 3.11.0 の AnnotateDispatches が K=0 の dot_general でゼロ除算する（#62、PR #65）

縮約次元のサイズが0の `dot_general`（`tensor<2x0xf32> . tensor<0x3xf32>`）を standalone の `iree-compile` に通しても SIGFPE で落ちる。原因は `mlir::iree_compiler::IREE::Flow::summarizeDispatchRegion`（`AnnotateDispatches.cpp`）内の整数0除算で、`iree-compile` にこのパスを無効化するフラグは無い。#53（PR #58）でこの整数0除算を `ARITHMETIC-ERROR` として捕まえ `IREE-COMPILE-ERROR` に変換するところまでは対応済みだったが、根本的にはコンパイルが通らない。最終的な対策は `dot-general` の `:emit` で、K=0 のときは `dot_general` を出さずに `stablehlo.constant dense<0.0>` を出す特殊化（PR #65）。上流（IREE）への報告は issue #73 に先送りしてある。

### 4.5 Lisp の非局所脱出が C++ フレームを飛び越え、in-process コンパイラが壊れる（#68、PR #69）

`nabla/iree` の medium スイート全体（複数の distinct な IREE コンパイルを重ねた後）で、in-process の `libIREECompiler.so`（`mlir::OpPassManager` の構築中）がメモリ破壊で落ち、SBCL プロセスごと `SB-SYS:MEMORY-FAULT-ERROR` で死ぬことがあった。`tests/iree/jit-test.lisp` の6テストだけを単独プロセスで動かすと問題なく、他の medium テストと同じプロセスで動かすと2/2で再現した。最小のコンパイル列は特定できておらず、原因は「Lisp の非局所脱出（コンディションの signal による unwind）が、コンパイラ内部の C++ フレームを飛び越えることで、C++ 側のデストラクタが呼ばれず内部状態が壊れる」という見立て（確証はない）。当面の対策は2段構え: (1) `tests/iree/jit-test.lisp` を独立した FiveAM スイート（`:nabla.isolated-medium`）にして別プロセスで実行しクラッシュを避ける、(2) PR #69 で、Pipeline 実行中に `ARITHMETIC-ERROR`（§4.4 のゼロ除算等）を捕まえたら、その場でコンパイラをこれ以上使わない「poisoned」状態にし、以後の呼び出しは（プロセスを巻き込むクラッシュより安全な）明示的なエラーとして拒否するようにした。根本原因の特定と修正は issue #68 に残っている。

### 4.6 check-it の往復（round-trip）検査が見つけた IR の等値性の落とし穴（#29、PR #44 / #48）

`print-graph` → `read-graph` → `print-graph` が同じテキストになることを検査する property-based testing と、その PR へのレビューで、2つの独立したバグが見つかった。1つは `read-graph` が `*package*` を何も `:use` しない専用パッケージに束縛して読むため、params に `T` のようなブール値が現れると `CL:T` と `eq` でない別のシンボルとして読まれ、round-trip の契約を壊すバグ（読んだ形全体を walk して正規化する `%normalize-graph-syntax-form` で修正）。もう1つは `print-graph` の内部で var 名を引く `gethash` が、見つからなかったときの挙動（デフォルトの `NIL`）を「値そのものが `NIL`」なのか「キーが無い」なのか区別せずに使っていたため、未定義参照を含む graph で `%nil` をそのまま印字してしまうバグ（`gethash` の第2値（found）を見るように修正）。どちらも「構造体やシンボルの同一性は、パッケージやハッシュ表の既定の読み方に暗黙に依存する」という落とし穴で、フェーズ2で grad の中間 graph を印字・比較するときにも同じ注意が要る。

### 4.7 stacked PR + squash merge のワークフロー

子 issue をおおむね1 PR 単位の stacked PR（下位ブランチに積む）で進めた。base ブランチが `main` でなくても CI が通る設定（`.github/workflows/ci.yml` の記述どおり）だったため、下位 PR がまだマージされていない段階でも上位 PR のレビューを進められた。マージ済みの下位ブランチを取り込むときの衝突は、squash merge 後の `main` を上位ブランチに `merge` または `rebase` せず取り込む（このリポジトリの運用は force-push・rebase を避ける方針、CLAUDE.md の Git の運用を参照）ことで、PR 内の途中コミットの乱れを気にせず進められた。`tests/primitives/registry-test.lisp` のように「p1〜p6 の全プリミティブが揃って初めて通る」テストを最後の子 PR に置く手法（§2.3）も、stacked PR の依存関係を機械的に検査する良い方法だった。

## 5. 正直な留保: mutation score 80% の検証は弱い

各子 issue の完了条件には「`tools/mutate/run.sh` でこの変更差分の mutation score が 80% 以上」という項目があるが、`tools/mutate` は次の理由でこの基準を強く検証できていない。

- 変異体は**トップレベルの定義1つにつき1個**しか作らない（複数箇所を独立に変異させない）
- 変異演算子は**4種類**（算術演算子の入れ替え・比較の境界・定数の置き換え・定数の置き換え、詳細は `tools/mutate/README.md` の「変異演算子」節と `.claude/skills/nabla-testing/references/mutation.md`）しかなく、条件分岐の削除・関数呼び出しの削除・境界値の反転など、他の mutation testing ツールが標準的に持つ演算子がない
- この結果、フェーズ1の各 PR は総変異体数が1〜7個程度にとどまり（PR #44 の実測は `total=2`、PR #48 は `total=5`）、`src/iree/` 配下の変更は総変異体数が0件（対象になる算術・比較・定数の形が無い）になることが常態化していた

つまり「mutation score 80%以上」は、対象がごく少数の変異体しかない場合の「たまたま全部殺せた」という弱い保証にしかなっていない。mutation runner 自体の演算子と粒度を増やす作業を issue #70 に切り出してある。フェーズ2で jvp / transpose ルールのような分岐が増える変換を書くときは、この issue が直るまで mutation score だけに頼らず、レビューと PBT の性質の充実度で品質を担保する必要がある。

（追記: issue #70 で対応済み。runner は変異をかけられるすべての箇所に1つずつ変異体を作り、演算子も条件の反転・式の削除・等価述語の入れ替え・`member` の要素の削除・文字列の置き換えを足して9種類になった。`src/eval.lisp`・`src/primitives/*.lisp`・`src/jit.lisp`・`src/stablehlo.lisp` での変異体の数と mutation score の実測は [docs/mutation-scores.md](mutation-scores.md) にある。）

## 6. 既知の制限と積み残し

- **#70**: mutation runner の演算子と粒度を増やし、「mutation score 80%」を意味のある基準にする（対応済み。§5 の追記と docs/mutation-scores.md 参照）
- **#71**: jit キャッシュの後始末と並行性を直す（`backend-unload` が GC 時に呼ばれない・ロックの粒度・`recompile` リスタートの再帰に上限が無い・`defjit` が `:static-args` 未対応、`src/jit.lisp` のコメント参照）
- **#72**: `f64` と `:i1` を `to-device` で扱えるようにし、op 対応表の全 dtype を `jit` で実行できるようにする（対応済み。f64 の exp / log / tanh は、それを含むモジュールだけ embedded ELF ではなく system library としてリンクすることで `jit` できるようにした（`ld.lld` が要る）。docs/stablehlo-ops.md 参照）
- **#73**: IREE 3.11.0 の `AnnotateDispatches` のゼロ除算（§4.4）を最小の再現にまとめ、上流に報告する
- **#74**: フェーズ1で残った配列 API とテスト支援の小さな制約を片づける（README の「既知の制約」に書いた、`dot` が `array`・`array` と `tracer`・`tracer` の2メソッドしか持たない等）
- **#12**: GPU（`cuda` ターゲット）での local/cuda 数値一致は、フェーズ0から引き続き環境に GPU が無く未測定（追記 2026-10-10: #12 の local/cuda の数値一致は Colab の Tesla T4 で確かめた。large スイートが `NABLA_REQUIRE_CUDA=1` で通った。結果は `docs/iree-build.md` の「実測（Colab Tesla T4）」）

## 7. フェーズ2（grad）への引き継ぎ

- **書き換えの挿入点**: `src/jit.lisp` の `%jit-cache-lookup-or-compile` が `graph-thunk`（trace）を呼んでから `emit-stablehlo`（emit）を呼ぶまでの間が、graph → graph の変換を挟む場所として最初から空けてある（jit.lisp 冒頭のコメント「trace -> graph -> emit -> compile -> load -> execute の6段階」を参照）。`grad` はここで `linearize` → transpose という JAX 方式の変換を書くか、最初は演算ごとに直接 vjp ルールを書く簡易版で始めてよい（設計タブ「変換の設計」）
- **defprimitive の拡張**: フェーズ1の19プリミティブは `:abstract-eval` / `:emit` / `:eager` の3点セットしか持たない。grad に対応させるには、各プリミティブに jvp ルール（順伝播の接線を計算）を追加し、線形なプリミティブ（`add`・`mul` の片側・`transpose`・`broadcast-in-dim`・`reduce-sum`・`dot-general` 等）には transpose ルールも追加する必要がある。`defprimitive` の構造体・マクロ自体（`src/primitive.lisp`）は新しいルール種別をキーに追加するだけで拡張できる形にしてある
- **候補の最初の一歩**: (1) `add`/`mul`/`neg`/`reduce-sum`/`transpose`/`reshape`/`broadcast-in-dim` のような線形演算から jvp + transpose を実装し、線形回帰程度の小さな関数で `grad` の end-to-end テストを先に書く。(2) `dot-general` の jvp/transpose は行列積の微分（転置と縮約次元の入れ替え）が絡むので、線形演算が一通り揃ってから着手する。(3) `exp`/`log`/`tanh`/`compare`/`select`/`convert`/`max`/`min` は非線形または非微分可能な演算なので、jvp のみ（transpose は無い、または `select` を使った勾配のマスクになる）
- **grad 実装順序**（設計タブに準拠）: jvp ルール → 線形プリミティブへの transpose ルール → `linearize`（jvp のトレースを非線形部分と線形部分に分割）→ `vjp`（線形部分を transpose）→ `grad`（vjp に単位余接線を流す）。高階微分や `vmap` との合成が必要になるまでは、演算ごとに直接 vjp ルールを書く簡易版でもよい
- **引き継げる公開 API**: README.md の「公開 API」節がそのまま前提にできる（dtype・aval・backend プロトコル・IR・トレーサ・jit・`eval-graph`）。`jit-compile-error` の `graph` / `eqn` フィールドは、grad が生成した graph のコンパイルが失敗したときの診断にも使える
- **§5 の留保を踏まえて**: grad のルールは分岐（線形/非線形の判定、transpose の方向）が増えるので、mutation runner が直るまでは PBT の性質（例: 中心差分・内積テストとの一致）を手厚くする
