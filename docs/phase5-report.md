# フェーズ5 報告書: Grain 相当のデータ供給層

フェーズ5（親 issue #278、「インデックスでレコードを取れるデータソース、決定的なサンプラー、map / filter / batch の変換、lparallel による並列ローダ、位置を保存して再開できるイテレータを積み、画像分類の学習を end-to-end で回して確かめる」）で作ったもの・得た知見をまとめる。形式は [phase4-report.md](phase4-report.md) に合わせてある。ロードマップ（フェーズ0〜5）はこのフェーズで終わるので、§7 は「フェーズ5の後」への引き継ぎにした。

## 1. 範囲と結論

#278 の完了条件に対する達成状況:

| 完了条件 | 判定 | 検証方法 |
| --- | --- | --- |
| 小さな CNN を `data-loader` のバッチで、`jit` した学習ステップで学習でき、損失が減る（#287） | 達成 | `examples/image-classification.lisp` と `tests/nn/image-classification-test.lisp`。合成画像 256 枚・batch 32・64 ステップで、IREE `local` の損失が 1.38 から 0.41 に、正解率が 25% から 78% に上がった（§4.1） |
| 同じシードからは、並列度に依らず同じバッチ列が出る（#280、#286） | 達成 | `tests/data/parallel-test.lisp` の PBT（`parallel-loader/batches-do-not-depend-on-the-number-of-workers` など。ワーカー数・先読みを変えたバッチ列が逐次とビット単位で一致）。CNN の学習でもワーカー 0 と 2 が一致する（eager の medium と IREE の medium） |
| 「モデル + オプティマイザ状態 + ローダの状態」を1つのチェックポイントに保存し、復元して続けた学習が、中断しなかった学習とビット単位で一致する（#285、#287） | 達成 | `tests/nn/checkpoint-test.lisp`（`iterator-state` の往復）、`tests/nn/image-classification-test.lisp`（エポックの境界をまたいで保存・復元した学習が、中断しない学習とビット単位で一致。IREE では「2 ワーカーで再開」と「0 ワーカーで中断なし」が一致）、`tests/nn/tutorial-test.lisp` |
| `.npy` のデータセットを読める（#282）。画像（PNG / JPEG）のデコードは #283 で扱いを決める | 達成 | `.npy`: `tests/data/npy-test.lisp`（numpy が書いたフィクスチャ `tests/fixtures/npy/` と、書いて読む往復の PBT）。画像: 別システム `nabla/data/image` に PNG（png-read）と JPEG（cl-jpeg）のデコードを入れた（§3.9）。`tests/data/image/image-test.lisp` が Pillow のデコード結果と比べる |
| README のチュートリアルがデータローダを使っている（#287） | 達成 | README の「チュートリアル」の手書きの `make-batch` を `array-source` → `data-loader` → イテレータに置き換え、チェックポイントにローダの状態を入れた。`tests/nn/tutorial-test.lisp` が README と `examples/tutorial.lisp` の一致・損失の減少・再開のビット一致を確かめる |
| 既定のテストスイート（small + medium）が CI で通る | スタックの CI で確かめる | このフェーズの PR は1本の線形スタックに積み、最後にまとめてマージする。スタックの PR を作ったときの CI が判定になる。手元の確認は §4.5 |

結論: 完了条件のうち、CI 以外はすべて満たした。子 issue 10 本（#279〜#288）のうち、#279 は PR #289 でマージ済みである。残りは、1 本の線形スタックに積んだ 13 本のブランチと、この報告書のブランチで進めた（大きな issue 4 本は 2 本ずつに分けた。§2）。実装の前に設計レビュー（Fable）で 9 つの判断を検証した。その中で、チェックポイントのパス名が子の並びで決まってしまう穴を見つけ、core に小さな変更（`defpytree` の子の名前）を先に入れた（§3.5）。このスタックでは IREE の回避策を新しく足していない。一方、cl-jpeg が `(safety 0)` でコンパイルされ、壊れた JPEG でプロセスが落ちることを見つけ、clone 時に書き換える暫定の対処を入れた（§3.9）。性能の積み残し #265 と #233 は別のスタックで調べ、どちらも emit は変えないと判断した（§3.11）。

## 2. 作ったもの

#279 以外はスタックのブランチ（PR 番号はスタックの作成時に決まる）。下から順に積んである。各ブランチの詳しい内容はコミットの本文にある。

### 2.1 nabla/data（データ供給の層）

- **#279（PR #289）**: `nabla/data` システムの新設。`data-source`（`source-length` / `source-item` の総称関数）と `array-source`（メモリ上の配列の木。各葉の先頭の軸がレコードの軸）。`data/depends-only-on-core` が asd の依存を検査する。
- **#280（`feat(data): add a deterministic index sampler`）**: `index-sampler`（`src/data/sampler.lisp`）。位置 k → `(values epoch index)` の純粋関数。シャッフルは SplitMix64 の鍵の均衡 Feistel 置換と cycle-walking（§3.2）。決まったシードの順序を既知の答えのテストで固定した。mutation 101/103（生き残り 1 は等価変異、1 はタイムアウト）。
- **#281a（`feat(data): add map and filter transforms`）**: `map-transform` / `filter-transform` / `apply-transforms`（`src/data/transform.lisp`）。
- **#281b（`feat(data): stack records into fixed-shape batches`）**: `batch-records` と `batch-error`（`src/data/batch.lisp`）。不揃いなレコードは treedef・形・element-type の別に報告する。
- **#285a（`feat(data): add a sequential data loader and iterator`）**: `data-loader` / `make-data-iterator` / `next-batch` / `epoch`（`src/data/loader.lisp`）。バッチはエポックをまたがず、端数は既定で捨てる。1 エポックからバッチが 1 つも作れなければ `:empty-epoch`（§3.3）。
- **#285b（`feat(data): save and restore the iterator state in checkpoints`）**: `iterator-state`（子 `key` / `epoch` / `position` / `version`）と `restore-iterator`（`src/data/iterator-state.lisp`。§3.4）。`nabla/nn/tests` が `nabla/data` に依存し、チェックポイントの往復を `tests/nn/checkpoint-test.lisp` で確かめる（`nabla/data` 本体の依存は core だけのまま）。
- **#284（`feat(data): add random map transforms keyed by sampler position`）**: `random-map-transform`。キーはシード・変換の通し番号・サンプラーの通し位置から SplitMix64 で作る（§3.6）。画像の拡張（左右反転・パディング付きの切り抜き）はテストと README の例に置き、公開 API にはしない。
- **#286a（`feat(data): load batches with lparallel workers in a fixed order`）**: `:num-workers`、`close-iterator`、`with-data-iterator`（`src/data/parallel.lisp`。§3.7）。
- **#286b（`feat(data): prefetch batches and measure the parallel loader`）**: `:prefetch`（既定 2）と `scripts/bench-data-loader.sh`、測定は [data-loader-parallel.md](data-loader-parallel.md)（§4.2）。
- **#282（`feat(data): read and write .npy files as a data source`）**: `read-npy` / `write-npy` / `npy-source`（`src/data/npy.lisp`。§3.8）。

### 2.2 core と nabla/nn

- **core（`feat(core): let defpytree name its children for checkpoint paths`）**: `defpytree` の flatten が 3 つ目の値で子の名前（文字列のリスト）を返せ、`tree-children` の 4 つ目の値になる。名前は treedef にも jit のキャッシュのキーにも入らない。nn のチェックポイントは名前があれば添字の代わりにそれをパス名に使う（§3.5）。
- **#287a（`feat(nn): train a small cnn from the data loader as the phase 5 acceptance test`）**: `examples/image-classification.lisp` と `tests/nn/image-classification-test.lisp`（§4.1）。`src/` は変えていない。
- **#287b（`docs: use the data loader in the readme tutorial`）**: README のチュートリアルと `examples/tutorial.lisp` をデータローダに置き換えた。

### 2.3 nabla/data/image

- **#283（`feat(data): decode png and jpeg images in nabla/data/image`）**: 別システム `nabla/data/image`。`read-image`（`(H W C)` の `(unsigned-byte 8)`）と `make-image-source`（レコードを取るときにデコードする）、`image-decode-error`。`scripts/setup-lisp-deps.sh` に png-read と cl-jpeg の固定コミットの clone と、cl-jpeg の safety の書き換えを足した（§3.9）。

### 2.4 別のスタック（性能。PR #295 → #296 → #297、CI は成功、未マージ）

フェーズ4の積み残しの性能の issue を、フェーズ5のスタックとは別の線で調べた。どちらも測定と判断の記録で、core の emit は変えていない（§3.11）。

- **#265**: 縮約軸が先頭にある `dot_general` の速さ。[dot-general-transpose.md](dot-general-transpose.md)（このスタックには無く、別のスタックのマージで入る）。
- **#233**: x86 ネイティブのランナー（GitHub Actions の ubuntu）で、scan の ys の競合を確かめる `workflow_dispatch` のワークフロー。結果は [iree-upstream-bugs.md](iree-upstream-bugs.md) の 7。

## 3. 設計上の判断

実装の前に、Advisor（Fable）が 9 つの判断（D1〜D9）を判定した。合格 4、条件付き合格 2、不合格 3 で、不合格の D5 は代案（§3.5）を採り、D8 / D9 はスタックから外した（§3.11）。スタックの順序も条件付き合格だったので、直した（§6）。以下は判断ごとの結論と理由。

### 3.1 nabla/data の依存（#279）

- **nabla のシステムのうち core（`nabla`）にだけ依存する**（`nabla/nn`・`nabla/iree`・`nabla/pjrt` を知らない。外部の lparallel はよい）。レコードとバッチは jit に渡す PyTree そのものなので、木の扱いは core の `tree-flatten` / `defpytree` を使い、data 層に別の定義を持たない。#278 の時点では「PyTree・PRNG・`defmodule` の状態を使うなら nn への依存が要る」が未決だった。再開状態を `defmodule` でなく名前付きの `defpytree` にした（§3.5）ので、core だけで済んだ。
- FFI を呼ばない純 Lisp の層にした。ローダはデバイスへの転送を持たず、`next-batch` はホストの配列の木を返す（§3.7）。

### 3.2 サンプラー: SplitMix64 + Feistel（#280、D1）

- **nabla の PRNG を使わない**。eager の PRNG は呼ぶたびに graph を組み、Threefry を Lisp で評価する（1 回に数十 µs）。位置ごとに引くサンプラーには重すぎる。SplitMix64 は数 ns で、data 層の中で閉じる。その代償として、JAX / Grain とはビット単位で一致しない。
- **エポックの置換を配列で持たず、位置ごとに計算する**。シードとエポックから SplitMix64 で 4 本のラウンド鍵を作り、2^(2h) ≥ n の領域の均衡 Feistel 置換を cycle-walking で [0, n) に収める（期待反復は 4 回未満）。n ≤ 1 は恒等。位置 → インデックスの純粋関数なので、再開状態は位置だけで済む。
- **偶奇の補正**: 均衡 Feistel は偶置換しか作らず、cycle-walking も偶奇を偏らせる（n = 2^(2h) − 1 ではほぼ奇置換だけになる）。鍵のもう 1 ビットで決まる互換（添字 0 と 1 の入れ替え）を最後にかけ、偶奇を半々にした。`index-sampler/orders-of-both-parities-occur` が守る。
- **算法の版**: 同じシード・エポック・位置からの順序は、版をまたいで変えない約束にした（チェックポイントから再開した列が変わらないため）。変えるときは `+index-sampler-algorithm-version+`（今は 1）を上げ、再開状態の `version` の検査（§3.4）で検出する。決まったシードの順序は既知の答えのテストで固定してある。random-map のキーの導出（§3.6）もこの版に含める。
- シャードは持たない（§5）。

### 3.3 変換・バッチ・エポック（#281、#285、D2）

- map / filter はレコード 1 つにかける純粋関数で、ワーカーから同時に呼ばれる。
- **バッチにまとめるのはローダの `:batch-size` の仕事**で、変換のリストには入れられない。「batch はパイプラインの最後に 1 つだけ」を規約でなく構造で守るため（D2 の条件）。公開するのはレコードのリストを積む関数 `batch-records` だけ。
- **バッチはエポックをまたがない**。端数は既定で捨て（`drop-remainder t`）、nil なら短いバッチにする。pad は持たない。jit はバッチの形ごとにコンパイルするので、形を揃える方が既定になる。
- **`:empty-epoch`**: `num-epochs nil` で filter が 1 エポック全部を落とすと、`next-batch` が戻らなくなる。1 エポックの全位置を調べてバッチが 1 つも作れなければ error にした（D2 の指摘）。
- 「1 エポックで各レコードがちょうど 1 回ずつ出る」は filter が無いときだけ成り立つ。テストの前提に書いてある。

### 3.4 再開状態は入力側の位置（#285、D2・D5）

- イテレータが持つのは「次に調べるサンプラーの位置」だけ。再開状態はこれをエポックとエポック内の位置に分けた **入力側の位置**（Grain と同じ）で、出力のバッチの番号ではない。サンプラーが位置の純粋関数で、変換が純粋なら、次が成り立つ。「この位置より前のレコードは、返したバッチに入ったか捨てられた」を満たす位置から再開すると、残りの列は中断しなかった場合と一致する。filter をはさんでも、並列でも、先読みしても崩れない（先読みした位置は状態に入れない）。
- `iterator-state` の葉は `key`（ローダのシードの写し、`:u32` の `(2)`）、`epoch` / `position` / `version`（rank 0 の `:i32`）。Lisp の整数はチェックポイントの葉にできない（フェーズ4、`:bad-leaf`）。レコードの個数とエポックの数が 2^31 未満なら収まる。
- `restore-iterator` は葉の型・形（`:bad-state`）、算法の版（`:version-mismatch`。別の版の状態から黙って別の順序で再開しない）、キー（`:key-mismatch`）、位置の範囲（`:state-out-of-range`）を検査する。データソース・変換が同じかは検査しない（利用者が同じ引数でローダを作る）。

### 3.5 defpytree の子の名前（core、D5）

- 草案は `iterator-state` を名前の無い `defpytree` の型にしていた。nn のチェックポイントは `defmodule` 以外の型の子を添字で名づけるので、パス名が `"2.0"` `"2.1"` になる。**パス名はファイル形式の一部**（`nabla.checkpoint_version "1"`）なので、子の並びを変えるだけで既存のチェックポイントが読めなくなる。Fable はこれを不合格とし、代案を出した。
- 採った代案: `defpytree` の flatten が 3 つ目の値で子の名前を返せるようにし、`tree-children` の 4 つ目の値で返す。既存の 3 値の呼び出しは壊れない。名前は treedef に入れない（jit のキャッシュのキーも変わらない）。nn は名前があればそれを、無ければ添字を使う。これで再開状態のパス名は `"2.key"` `"2.epoch"` `"2.position"` `"2.version"` になった。名前の個数違い・空・`.` を含む・重複は `pytree-error`。
- data 層が nn の `defmodule` を使えない（§3.1）ことと、パス名を安定させることを、core の約 70 行の変更で両立できた。

### 3.6 random-map のキー（#284）

- 変換関数は `(record key)` を受け、key は `prng-key` と同じ形の `:u32` の `(2)`。利用者は `nb:uniform` / `nb:split` をそのまま使える（eager）。
- キーは「シード」「変換のリスト内の通し番号 j」「サンプラーの通し位置 k」だけで決まる。状態 mix64(mix64(シード) xor mix64(j)) の SplitMix64 の列の k 番目を上位 / 下位 32 ビットに割る。シードを先に攪拌するので、シードの差と位置の差が打ち消し合わない。**エポックとインデックスでなく通し位置 k で決める**ので、同じレコードでもエポックが違えば別の拡張になる（Grain と同じ）。並列度・処理の順序・再開に依らない。
- `nabla:fold-in` は使わない。eager は位置ごとに graph を組むので遅い（§3.2 と同じ理由）。

### 3.7 並列ローダ（#286、D6）

- **イテレータごとに専用の lparallel のカーネルを持つ**（利用者の `lparallel:*kernel*` とは共有しない）。`make-kernel` の `:bindings` で **`lparallel:*debug-tasks-p*` を nil にする**。既定の t では、ワーカーの中の error がワーカーのスレッドでデバッガを呼び、非対話のテストと CI が止まる。nil なら `force` した側に元のコンディションが届く。
- 1 バッチの位置をワーカー数個の連続区間に分け、区間ごとの `future` を **作った順に `force`** する（`receive-result` は完了順なので使わない）。組み立て（バッチの切り方・エポックの境界・端数）は逐次と同じコードを通るので、バッチ列はワーカー数・先読みに依らずビット単位で同じになる。`:num-workers 0` はスレッドを作らない逐次の経路で、一致のテストの基準になる。
- 先読みは「返したバッチの先 `prefetch` 個ぶんの位置」を投げておく。要求と食い違えば待ち行列を捨てて投げ直す。再開状態は返したバッチの位置なので、先読みに影響されない。
- **ワーカーは `device-array` を作らない**。backend の `to-device` は呼び出し側のスレッドで同期して走るので、ローダの中で転送しても重ならない。重なるのはワーカーの読み込みと、呼び出し側の転送・学習ステップ（§4.2）。
- 片付けは `close-iterator` と `with-data-iterator`（`end-kernel :wait t`）。閉じないイテレータのスレッドは GC では止まらない（§5）。イテレータそのものはスレッド安全でない。

### 3.8 .npy は全体をメモリに読む（#282、D3）

- 遅延読み込み（mmap）は測ってから決める。CNN とチュートリアルの規模では要らない。
- bool は bit 配列でなく `(unsigned-byte 8)` の 0 / 1 で返す。葉を型特化した配列に揃え、後の map で f32 / i32 に直す経路を u8 と同じにするため。
- ヘッダの dict は 3 つのキーだけの最小のパーサ。Fortran 順（1 を超える軸が 2 つ以上）・ビッグエンディアン・object / complex / 構造化 dtype は `npy-error`。形とファイルの大きさを、配列を確保する前に照合する。
- `write-npy` は numpy と同じヘッダのバイト列を書き、`with-atomic-output-file` で置き換える。

### 3.9 画像のデコードは別システム（#283、D4）

- **`nabla/data/image` を別システムにした**。`nabla/data` を依存の軽い純 Lisp の層のまま保ち、画像のライブラリの調達で詰まってもスタックの残りを止めないため。受け入れテスト（#287）は画像ファイルを使わず npy で回す。
- ライブラリは png-read（依存は chipz・iterate・babel）と cl-jpeg（依存なし）。どちらも純 Lisp。opticl と pngload は依存が多いので採らなかった。自前の PNG のデコーダは inflate だけで 300 行ほどになるので選ばなかった。chipz・iterate・babel は apt にある。png-read と cl-jpeg は Ubuntu noble に無いので、固定コミットの git clone にした。
- **cl-jpeg の safety**: cl-jpeg は全体を `(safety 0)` でコンパイルするので、壊れた JPEG で範囲外のメモリに触れてプロセスごと落ちる。fork で直すまでの暫定として、`setup-lisp-deps.sh` が clone の直後に `*optimize*` を `(safety 2)` に書き換える。safety 1 は cl-jpeg の `(speed 3)` より低く、SBCL が型の検査を弱めるため、SBCL 2.2.9 の x86-64 ではなお落ちた。書き換えはスクリプトのインラインの置換なので、CI のキャッシュのキーに入る。置換はべき等で、上流の行が変われば error で止まる。`read-image` は JPEG をデコードする前に毎回 `jpeg::*optimize*` の safety を確かめ、2 未満なら `setup-lisp-deps.sh` を挙げる error にする。ASDF のメソッドで safety を制限する案は、大域のメソッドを足すので採らなかった。
- JPEG の画素は Pillow と成分ごとに最大 1 違う（逆 DCT の丸めと色差の補間）。PNG はちょうど一致する。

### 3.10 受け入れテストのデータセット（#287、D7）

- CI がネットワークに頼らないよう、合成画像（16x16 のグレースケール、4 クラス: 縦線・横線・斜線・十字 + 雑音）を実行時に決定的に作る。画像はプロセスごとの一時ディレクトリに `write-npy` で書き、`npy-source` で読む。npy の実地の利用者にもなる。
- ローダ・再開・ワーカー数の決定性は eager の medium でも確かめ、IREE の medium は「例の `main` が学習し、再開がビット単位で一致する」1 本に絞った。手元の Mac に IREE が無く、IREE のテストは VM と CI でしか走らないため。
- jit する学習ステップの入力は 21 個（パラメータ 6 + mu 6 + nu 6 + count 1 + バッチ 2）で、local-task の上限 160 に収まる。small のテストが確かめる。

### 3.11 スタックに入れなかったもの（#265、#233、D8・D9）

- **#265**（D8、不合格）: IREE の StableHLO の入力の前処理は `dot_general` を transpose + 正規形に書き換えるので、nabla が transpose を明示しても同じ形に畳まれる見込みが高い、と Fable が指摘した。別のスタックで測った結果、明示の transpose は IREE が matmul に畳むので効かなかった。barrier を挟んで畳まれないようにすると、iree-local の large の MLP は 26% 速くなった。ただし iree-local の small の MLP は 40%、pjrt-cpu の large の MLP は 45% 遅くなった（x86_64 の VM の Rosetta で測った比。[dot-general-transpose.md](dot-general-transpose.md)、別のスタックにある）。core の emit は全 backend の StableHLO とディスクキャッシュのキーを変えるので、変えない。IREE の backend だけで書き換える候補は後続（§5）。
- **#233**（D9、不合格）: 安全性が確かめられていない emit を opt-in で入れる案は見送った。静かに 0 の行が混ざる正しさの事故になるため。代わりに x86 ネイティブのランナーで race を確かめるワークフローを足した。候補 (b) joint-barrier は 1 回目の run では 200000 回中 0 回の不一致だったが、一般化した形（y_t が派生値）では 0.40% の不一致が出た。2 回目の run では、(b) 自身も 200000 回中 1 回の不一致が出た。emit は変えず、上流の修正を待つ。

## 4. 測定

### 4.1 CNN の end-to-end（#287）

構成: 合成画像 256 枚（16x16x1、4 クラス）、batch 32、シャッフル、u8 → f32 と one-hot の map、左右反転とパディング 2 の切り抜きの random-map。CNN は `conv` 3x3 stride 2 を 2 段 + relu、空間の平均、`dense`（パラメータの葉 6 個）。softmax 交差エントロピー + `adam 0.03`。64 ステップ（8 エポック）で、32 ステップ目に「モデル + オプティマイザ状態 + ローダの状態」を 1 つの safetensors に保存し、復元して続ける。

| 測ったもの | 結果（IREE local、x86_64 の VM） |
| --- | --- |
| 損失 | 1.38 → 0.41 |
| 訓練データの正解率 | 25% → 78% |
| 保存・復元をはさんだ学習（2 ワーカー） | 中断しない学習（0 ワーカー）とビット単位で一致 |

測定した VM は OrbStack の x86_64 の VM（Rosetta、SBCL 2.2.9）。テストの閾値は「損失が下がる」「正解率が 50% を超える」で、ちょうどの値には頼らない。

### 4.2 並列ローダ（#286）

詳細は [data-loader-parallel.md](data-loader-parallel.md)。重い map（純 Lisp の 3x3 の平均を 40 回。1 レコード数 ms）をかけた 1024 レコード（batch 32）の 1 エポックを Apple M1（高性能 4 + 高効率 4 コア、SBCL 2.6.9）で測った。

| 条件 | 逐次 | ワーカー 8・先読み 2 | 比 |
| --- | ---: | ---: | ---: |
| 呼び出し側の計算なし | 2.90 秒 | 0.59 秒 | 4.9 倍 |
| バッチごとに 20 ms の計算（学習ステップの代わり） | 3.54 秒 | 0.75 秒 | 4.7 倍 |

- 20 ms の計算をはさむと、先読みなし（ワーカー 8）は 1.32 秒、先読み 2 は 0.69〜0.75 秒で、ほぼ計算の時間（32 × 20 ms = 0.64 秒）だけになった。先読みが読み込みと計算を重ねている。
- 4 ワーカーを超えると伸びが鈍る（M1 の高効率コア）。変換が軽いときは future と待ち合わせの費用が目立ち、ワーカーで速くならないことがある（測っていない）。
- Linux の x86 では測っていない。手元の x86 は Rosetta の VM しか無く、スレッドの性能の参考にならないため（§5）。

### 4.3 そのほかの測定

- **画像のデコード**（#283）: 224×224 の画像 1 枚が JPEG 約 5 ms、PNG 約 8 ms（Apple M 系、SBCL 2.6.9）。ワーカーで並列に読む前提の重さ。
- **eager の PRNG**: 1 回に数十 µs（README）。random-map の中で 1 つのキーから `nb:uniform` で必要な個数をまとめて引くよう README に書いた。npy の小さなレコードでは、拡張の費用の大半が PRNG になりうる。
- **sampler**: 位置ごとに Feistel を計算するので、エポックの置換を持たない（メモリは n に依らない）。時間は測っていない（レコードの読み込みと変換に比べて小さい）。

### 4.4 mutation testing

各ブランチで、変更した行にかけた。数は作業ログに残っている最後の実行のもの。`src/data/sampler.lisp` は 103 体中 101 体を殺し、残りは等価変異 1 体（n = 0 の位置の上限。先に k ≥ 0 を検査するので、メッセージしか変わらない）とタイムアウト 1 体。`defpytree` の子の名前（`src/pytree.lisp`）は 16/16、`src/data/iterator-state.lisp` は 53 体中 52 体（1 体は除外）、`src/data/batch.lisp` は 22 体中 20 体。ローダ（`src/data/loader.lisp`）は 20 体中 14 体を殺し、残りの 6 体はタイムアウトで、生き残りは無い。

### 4.5 既定のテストスイート

- 手元の Mac（arm64、SBCL 2.6.9）には IREE / PJRT が無いので、IREE / PJRT の medium は自動でスキップされる。各ブランチで `scripts/run-tests.sh` を流し、Mac 固有の既知の失敗（safetensors の `/private/var` のパス、子プロセスの sbcl、シグナルのテスト、ffi-support）以外が無いことを確かめた。この報告書のブランチ（スタックの先端）では、small 148450 checks がすべて通り、medium 754 checks のうち失敗は既知の 5 テスト・6 件、スキップは 350 件だった。
- IREE / PJRT のテストは OrbStack の x86_64 の VM（Rosetta、SBCL 2.2.9）で `NABLA_REQUIRE_IREE=1 NABLA_REQUIRE_PJRT=1` で流した。VM では `main` の時点で 9 テスト・17 件が落ちる（attention の jit で NaN、PJRT の PBT、input-limit の既知のバグ、float trap）。Rosetta に由来すると見られ、CI の ubuntu では通る。各ブランチで、これ以外の失敗が増えないことを確かめた。
- CI（ubuntu-24.04、8 シャード）での確認は、スタックの PR を作ったときに行う。

## 5. 既知の制限と積み残し

issue はまだ起こしていない（スタックのマージ後に起こす）。

- **cl-jpeg の fork**: safety の書き換え（§3.9）は暫定。fork して `*optimize*` を直し、`setup-lisp-deps.sh` の置換と `%check-cl-jpeg-safety` を外す。fork はリポジトリのオーナーの作業。
- **#265 の IREE だけの候補**: 縮約軸が先頭の `dot_general` を IREE の backend だけで transpose + barrier に書き換える案（iree-local の large の MLP で 26% 速いが、small の MLP では 40% 遅い）。大きさで切り替える条件を決めてから。core は IREE の名前を知らない約束に注意する。
- **#233 は上流の修正待ち**（§3.11）。上流への報告は #243 と一緒にリポジトリのオーナーが起票する。
- **`iterator-state` のひな形を、イテレータを開かずに作る関数が無い**。今は `(iterator-state (make-data-iterator loader))` で作る（ワーカー数 0 ならスレッドは作らない）。`(loader-state-template loader)` のような関数があると、復元の手順が読みやすくなる。
- **閉じないイテレータのスレッドは残る**。ワーカーのスレッドは GC では止まらない（イテレータに finalizer を付けていない）。`with-data-iterator` か `close-iterator` で必ず閉じる、と README に書いてある。
- **シャード（複数プロセスでの分割）が無い**。サンプラーは位置 → インデックスの関数なので、「プロセス i は位置 i, i + P, ... を引く」形で後から足せる。複数のホストで学習する段階で。
- **npy の遅延読み込み（mmap）が無い**。データセットがヒープに入らない段階で。チェックポイントの遅延読み込み（#262）と同じく、測ってから決める。
- **u8・i8・i64・bool の dtype が nabla に無い**。npy や画像の u8 は、jit に渡す前にローダの map で f32 / i32 に直す。画素を u8 のまま device に送れれば転送量が 4 分の 1 になる。core に dtype を足すのは、転送が律速になったと測ってから。
- **ローダの速さを Linux の x86 と CI で測っていない**（§4.2）。`scripts/bench-data-loader.sh` は CI のランナーでも流せる。
- **再開のときに、データソースと変換が同じかは検査しない**（§3.4）。利用者が同じ引数でローダを作る約束。
- **画像**: 16 ビットの PNG・CMYK の JPEG・可逆 JPEG は読まない。巨大な寸法を書いたファイルは、その大きさの配列を確保しようとする（上限を設けていない）。
- **pad が無い**。端数のバッチは捨てるか短いバッチにする。短いバッチは jit を再コンパイルさせる。
- **GPU**: CNN の学習とローダは CUDA で測っていない。手順はフェーズ2〜4と同じ（`scripts/colab-gpu-check.sh`）。
- **VM の失敗**: Rosetta の VM で `main` から落ちる 9 テスト（§4.5）は、CI の ubuntu では通るので、調べていない。

## 6. 進め方の知見（プロセス）

- **実装の前の設計レビュー**: フェーズ4は PR ごとに Advisor（Fable）が検証したが、フェーズ5は実装の前に、設計の判断 9 つとスタックの順序をまとめて判定してもらった。効いたのは 3 点。(1) チェックポイントのパス名がファイル形式になる穴（D5）を、実装する前に core の小さな変更で塞げた。(2) `*debug-tasks-p*` の既定・`:empty-epoch`・偶奇の偏りのような、実装してから見つけると高くつく落とし穴を先に挙げた。(3) 依存の向きからスタックの順序を 2 か所直し（#284 を #285 の後、#283 を #287 の後）、大きな issue 4 本を 2 本ずつに分けた。
- **1 本の線形スタックに積んで最後にまとめてマージする**。フェーズ4の最大のコストだった `nabla.asd`・パッケージの定義・README・CLAUDE.md の衝突は、ブランチを順に積むことで 1 回ずつの追記になった。代わりに、下のブランチを直すと上を全部積み直す必要がある。
- **IREE のある環境が手元に無い**: Mac には IREE / PJRT が無く、x86_64 の VM は Rosetta で、`main` の時点で 17 件が落ちる。eager の medium で決定性を確かめ、IREE のテストを 1 本に絞った（§3.10）のは、この制約への対処でもある。VM の失敗は `main` との差で判定した。
- **ライブラリの安全性はファズで確かめる**: cl-jpeg の `(safety 0)` は、壊れた JPEG を与えるテストで見つけた（`image/jpeg-with-changed-bytes-stays-in-bounds`）。safety 1 で足りるかは処理系とアーキテクチャで変わった（arm64 の SBCL 2.6.9 では足り、x86-64 の 2.2.9 では落ちた）。外部のライブラリを入れるときは、壊れた入力のテストを両方の環境で流す。
- **性能の調査は別の線で**: #265 と #233 はフェーズ5の機能と独立なので、別のスタック（PR #295〜#297）にした。どちらも「測った結果、emit は変えない」で、判断の根拠を docs に残した。

## 7. フェーズ5の後への引き継ぎ

ロードマップ（フェーズ0〜5）はこのフェーズで終わる。次に何をするかは計画の Artifact で決める。ここでは引き継ぐ事実をまとめる。

- **学習のループの形**: `data-loader` → `with-data-iterator`（`:num-workers` / `:prefetch`）→ `next-batch` のホストの配列の木 → `jit` した学習ステップ（パラメータ・オプティマイザ状態・バッチ）。保存は `(list model opt-state (iterator-state it))` を 1 つのチェックポイントに入れ、`restore-iterator` で続ける。手本は README のチュートリアルと `examples/image-classification.lisp`。
- **決定性の約束**: バッチ列とデータ拡張は、ローダのシードとサンプラーの位置だけで決まる（ワーカー数・先読み・処理の順序・再開に依らない）。シャッフルや random-map のキーの導出を変えるときは、`+index-sampler-algorithm-version+` を上げる。チェックポイントのパス名は `defpytree` の子の名前で決まるので、`iterator-state` の子の名前も形式の一部。
- **大きなデータ・モデルに進むとき**: npy とチェックポイントの遅延読み込み（#262）、シャード、u8 の dtype、ローダの速さの Linux での測定（§5）。どれも測ってから決める。
- **IREE の制限は変わらない**: 学習ステップの入力は葉の個数で、jit は上限を超えると dtype ごとに連結する（#242）。上流への報告（#243、#233 の 7）はリポジトリのオーナーの作業として残っている。#265 の IREE の backend だけの書き換えも候補として残っている（§5）。
- **GPU**: フェーズ4・5の機能（Transformer・CNN・ローダ）は CUDA で測っていない。
