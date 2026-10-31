# チェックポイントのピークメモリと世代管理（issue #244）

`save-checkpoint` / `load-checkpoint` が大きなファイルでどれだけメモリを使うかを測り、葉ごとの遅延読み込みと世代管理（最新 n 個を残す）が要るかを判断した記録。#244 で測り、#262 で葉ごとの読み書きにしてから測り直した（§2、§4）。

## 結論

- **ピークは、読み込みも保存もファイルの約 1 倍（木の分を含む）**（#262 から。それより前は読み込みで約 2 倍、保存で約 3 倍）。`read-safetensors` はヘッダを読んでからテンソルごとに配列へ読み、`write-safetensors` はヘッダを書いてから葉を 1 つずつ書く。どちらも上乗せするのはヘッダと 64 KiB の作業用バッファだけ。ヒープが 1 GiB のとき（この測定環境の既定。`(sb-ext:dynamic-space-size)` で確認）、同じ大きさの葉 4 枚（`few`）なら読み込みも保存も 896 MiB まで通る（1024 MiB は木そのものがヒープに入らない）。小さな葉 4096 枚を含む `mixed` では、読み込みが 768 MiB まで、保存が 512 MiB まで通る（保存の 640 MiB は、保存の前に木を作って GC するだけで枯渇する。上限を決めているのは木そのもの）。#244 の時点の境界は、読み込みが 448 MiB が通り 512 MiB で枯渇、保存が 256 MiB が通り 320 MiB で枯渇だった。
- **葉ごとの遅延読み込み（必要な葉だけ読む）は作らない**。ピークが約 1 倍になったので、残る上限は結果の木そのもの。フェーズ5の状態（データパイプラインの位置・PRNG キー・小さな配列）は KB 級で、フェーズ4のモデルも MB 級（§4）。
- **世代管理は要らない**。保存はアトミックなので、最新の 1 個は常に壊れない。「最新 n 個を残す」は学習ループの方針（フェーズ5以降）で、呼び出し側が数行で書ける（§5）。
- データパイプラインの状態は、**Lisp の整数ではなく配列で持つ**。子に名前を付けた `defpytree` の型（`nabla/data` は nn に依存できないので `defmodule` は使えない）にすれば、モデル + オプティマイザ状態と同じ木で往復する（§3）。

## 1. 測定環境

| 項目 | 値 |
| --- | --- |
| OS / CPU | macOS（Darwin 27.0.0）、arm64、メモリ 16 GiB |
| 処理系 | SBCL 2.6.7（#244）、SBCL 2.6.9（#262 の前後の測定。#262 の前の値は 2.6.7 の値と ±5 MiB で一致した） |
| `dynamic-space-size` | この環境の既定の 1 GiB（`(sb-ext:dynamic-space-size)` = 1073741824。他の環境の既定は未確認）と、8 GiB を指定した場合 |
| 対象 | f32 のテンソルだけの木（リスト）。`few` = 同じ大きさの葉 4 枚、`mixed` = 大きな葉 4 枚（全体の 3/4）+ 小さな葉 4096 枚（残り 1/4） |
| 合計の大きさ | 64 / 256 / 1024 MiB |

再現: `scripts/measure-checkpoint-memory.sh`（ケースごとに別の SBCL プロセス。`scripts/measure-checkpoint-memory.lisp` が 1 ケースを測る）。

測り方:

- 指標は **最大 RSS**（`getrusage` の `ru_maxrss`）。`dynamic-usage` を GC のたびに拾う方法（スクリプトの `:heap-peak-mb`）は、GC の間の山を逃すので参考値にとどめる。
- 「倍率」は **ピーク RSS ÷ ファイルの大きさ**。save では木そのもの（値が 0 の f32 配列）を含み、load はひな形を `aval` にして木の分を持たない。基準の RSS（木を作って full GC した直後）は load（ひな形が `aval`）で約 60 MiB、256 MiB 以上では無視できる。save の基準は、値が 0 の配列が遅延ゼロページで RSS にほとんど載らないため小さく出る（`:rss-base-mb` が 58〜148）が、保存中に木のページがタッチされて RSS に乗るので、save の倍率は基準を引かずピーク RSS で見る。
- `ru_maxrss` はプロセスの一生の最大で、nabla/nn をロードする山が約 300 MiB あるので、**64 MiB のケースの RSS は読めない**（`dynamic-space-size` が 1 GiB のときは山が小さく読める）。倍率は 256 MiB 以上で見る。
- 大きなケース（約 800 MiB 以上）では、macOS のメモリ圧縮で RSS が実際より小さく出ることがある（ほかのプロセスが動いているとき。896 MiB の `few` は RSS が 573 / 542 MiB だったが、`:heap-peak-mb` は 945 MiB）。境界の判定は「通ったか」で見て、ピークは `:heap-peak-mb` でも確かめる。

## 2. 結果

最大 RSS（MiB）。かっこ内はファイルの大きさに対する倍率（基準の約 60 MiB を含む）。「#262 の前」は同じマシン・SBCL 2.6.9 で main を測り直した値（#244 の 2.6.7 の値とほぼ同じ）。

| 操作 | 構成 | 合計 | 1 GiB（#262 の後） | 8 GiB（#262 の後） | 1 GiB（#262 の前） | 8 GiB（#262 の前） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| load | few | 256 | 315（1.2x） | 319（1.2x） | 571（2.2x） | 575（2.2x） |
| load | mixed | 256 | 338（1.3x） | 353（1.4x） | 601（2.3x） | 603（2.4x） |
| load | few | 1024 | **ヒープ枯渇** | 1088（1.1x） | **ヒープ枯渇** | 1958（1.9x）※ |
| load | mixed | 1024 | **ヒープ枯渇** | 1265（1.2x） | **ヒープ枯渇** | 2270（2.2x） |
| save | few | 256 | 315（1.2x） | 319（1.2x） | 827（3.2x） | 831（3.2x） |
| save | mixed | 256 | 341（1.3x） | 346（1.4x） | 936（3.7x） | 879（3.4x） |
| save | few | 1024 | **ヒープ枯渇（SBCL が落ちる）** | 1086（1.1x） | **ヒープ枯渇（SBCL が落ちる）** | 2193（2.1x）※ |
| save | mixed | 1024 | **ヒープ枯渇（SBCL が落ちる）** | 1107（1.1x） | **ヒープ枯渇（SBCL が落ちる）** | 2225（2.2x）※ |

※ #244 では 2111 / 3135 / 3427 MiB。差はメモリ圧縮によるもので（§1）、`:heap-peak-mb` は 1841 / 3121 / 3125 MiB だった。

1 GiB のヒープでの境界（#262 の後。448 / 512 / 640 / 768 / 896 MiB を測った）:

| 操作 | `few` | `mixed` |
| --- | --- | --- |
| load | 896 MiB まで通る（1024 は枯渇） | 768 MiB まで通る（896 で枯渇） |
| save | 896 MiB まで通る（1024 は枯渇） | 512 MiB まで通る（640 で枯渇） |

1024 MiB は結果の木（load）・保存する木（save）そのものが 1 GiB のヒープに入らない。`mixed` の save の 640 MiB は、保存を呼ばずに木を作って full GC するだけでも同じく GC の最中に枯渇した（小さな葉 4096 枚を GC がコピーする余裕が無い）ので、上限を決めているのは保存ではなく木そのもの。#262 の前の境界は、load が 448 MiB（ピーク 954）まで通り 512 で枯渇、save が 256 MiB（827）が通り 320 MiB から枯渇だった（`few`）。

読み取り:

- **load は約 1 倍**。`read-safetensors` はヘッダ（JSON）だけを先に読み、ファイルの長さと照らしてテンソルの範囲を検査してから、テンソルごとに `file-position` で位置を合わせ、64 KiB の作業用バッファを通して配列の記憶域へ読む。残るのは結果の配列（1 倍）だけ。#262 の前はファイル全体を 1 本の octet 配列に読み（1 倍）、そこから葉ごとにコピーしていた（もう 1 倍）。
- **save は木そのもの + ヘッダ**。`write-safetensors` はヘッダを書いてから、葉ごとに作業用バッファを通して書く（要素型 `t` でない simple-array は記憶域から直接コピーする。調整可能・displaced の配列だけは 1 枚ずつ simple-array にコピーする）。#262 の前は木から作る葉ごとの octet 列（1 倍）と、それを 1 本にまとめた `encode-safetensors` の結果（1 倍）を持っていた。
- 小さな葉が 4096 枚あっても（`mixed`）倍率はほとんど変わらない。葉の数は支配的でない（ただし GC の余裕には効く。上の境界）。
- 所要時間は 1024 MiB・8 GiB のヒープで load 0.2〜0.4 秒、save 0.4〜0.6 秒（#262 の前は load 0.45 秒、save 1.6〜1.8 秒）。ディスクのキャッシュに載る範囲。
- 確保量は medium テスト `safetensors/file-io-allocates-about-one-file` が守る（8 MiB のファイルで、`write-safetensors` の確保がファイルの 1/4 未満、`read-safetensors` が 1.25 倍未満。`sb-ext:get-bytes-consed` で測る）。

## 3. データパイプラインの状態

フェーズ5のデータパイプライン（Grain 相当）の状態は、**配列だけの、子に名前を付けた `defpytree` の型**にして、モデルとオプティマイザの状態と同じ木に入れる。`nabla/data` は core にしか依存できない（nn の `defmodule` を使えない）ので、`defpytree` の `flatten` が3つ目の値で子の名前を返す（#285 の前提として core に足した）。名前を返さない `defpytree` の型の子は添字のパス名（`"2.0"`）になり、フィールドの並びを変えるだけで既存のチェックポイントが読めなくなるため、名前を付ける。

```lisp
(defstruct pipeline-state epoch position shuffle-key buffer)   ; どれも配列
(defpytree pipeline-state
  :flatten (lambda (s)
             (values (list (pipeline-state-epoch s)          ; rank 0 の :i32 配列
                           (pipeline-state-position s)       ; バッチ内の位置。rank 0 の :i32 配列
                           (pipeline-state-shuffle-key s)    ; (prng-key seed) か split の1行。:u32 の (2)
                           (pipeline-state-buffer s))        ; 小さな配列
                     nil
                     '("epoch" "position" "shuffle-key" "buffer")))  ; 子の名前（パス名になる。treedef には入らない）
  :unflatten (lambda (aux children)
               (declare (ignore aux))
               (destructuring-bind (epoch position shuffle-key buffer) children
                 (make-pipeline-state :epoch epoch :position position :shuffle-key shuffle-key
                                      :buffer buffer))))
(save-checkpoint path (list model opt-state pipeline))   ; "2.epoch" "2.shuffle-key" ...
```

- **Lisp の整数・実数を葉にすると `:bad-leaf`**（保存でもひな形でも）。これは意図した挙動で、変えない。理由: 葉は必ず dtype を持つ配列で、`tree-unflatten` で復元したときに型が決まるため。オプティマイザの `count`（rank 0 の `:i32` の配列。Lisp の整数にすると `jit` が毎ステップ再コンパイルする）と同じ方針で、**状態は配列で持つ**。
- PRNG のキーは `prng-key` が作る `:u32` の `(2)` の配列で、そのまま葉になる（`split` の結果の 1 行も同じ形）。往復したキーを `split` した結果は、元のキーから `split` した結果とビット単位で同じ。
- 文字列で足りる補助情報（データセットのパスや版など）は、`save-checkpoint` の `metadata` に入れてもよい。ただし配列の状態をメタデータに書き込むと形・dtype の照合が効かなくなるので、状態は葉にする。
- テスト: `tests/nn/checkpoint-test.lisp` の `checkpoint/named-registered-node-round-trips-under-its-names`（medium。子の名前を返す `defpytree` の節が `"親.名前"` のエントリで往復する）、`checkpoint/pipeline-state-in-the-train-state-tree-round-trips`（medium。上の形の名前つきの `defpytree` の状態を含む、モデル + オプティマイザ状態 + パイプラインの状態の木を PBT で往復し、整数の端（i32 の最小・最大）、符号付き・符号なしのシードから作ったキー、キーの `split` までビット単位で確かめる）と `checkpoint/lisp-integers-are-not-leaves`（Lisp の整数は `:bad-leaf`）。
- #244 ではコードは変えていない（上のテストは今の挙動を固定する）。#244 の時点では配列だけの `defmodule` にする案だったが、`nabla/data` が nn に依存できないので、#285 の前提として `defpytree` の子の名前を core に足し、それを使う形に改めた。

## 4. 葉ごとの遅延読み込みは要るか

**葉ごとの読み書きは #262 で実装した。必要な葉だけ読む形は作らない**。

- #244 の時点では、ヒープ 1 GiB で save が 256 MiB（約 25M パラメータ + Adam の状態）までしか通らず、GPT-2 small（124M パラメータ、f32 で約 500 MiB、Adam の状態を含めて 1.5 GiB）は 8 GiB のヒープでも save のピークが約 4.5 GiB になる見込みだった。
- #262 で、読み込みはヘッダを読んでからテンソルごとに配列へ読む形に、保存はヘッダを書いてから葉を 1 つずつ書く形にした（ファイル形式は変えない。`encode-safetensors` のバイト列とのビット単位の一致、Python が書いたファイルの読み込み、壊れたファイルの検出は今までのテストと、`safetensors/written-file-is-the-encoding`・`safetensors/read-agrees-with-decode-on-corrupted-files` が守る）。ピークは木そのもの + ヘッダ + 64 KiB になり、上限は木がヒープに入るかどうかで決まる。GPT-2 small の 1.5 GiB の状態は、それが入るヒープ（例 `--dynamic-space-size 4GB`）で読み書きできる。
- 「必要な葉だけ読む」形（ひな形の葉だけ、パイプラインの状態だけを先に読む）は、今のところ使い道が無いので作らない。読み込みはテンソルごとに `file-position` で位置を合わせているので、足すときは `%read-safetensors-from` に読む名前の集合を渡せば済む。
- 回避策は今までどおり `--dynamic-space-size` を上げること（利用者が SBCL の起動オプションで指定する）。

## 5. 世代管理（最新 n 個を残す）は要るか

**要らない**（今は作らない）。

- 「保存の途中で落ちても前のチェックポイントを失わない」は、一時ファイル + rename（`write-safetensors`）で既に満たしている。世代管理が守るのは、それとは別の失敗（保存したチェックポイントそのものが学習の発散後に使えない、ディスクの破損、評価値が良かった時点に戻したい）。
- 「何世代残すか」「評価値で選ぶか」「いつ消すか」は学習ループの方針で、ライブラリの `save-checkpoint` に持たせると、`best` を残す・n 世代 + 最良 1 個を残す、のような要望ごとに引数が増える。呼び出し側が保存先のパス名にステップ数を含めて（`(format nil "ckpt-~8,'0D.safetensors" step)`）、保存後に古いものを消せば足りる（`uiop:directory-files` で数行）。
- フェーズ5の再開（データパイプラインの位置を含む）に必要なのは最新の 1 個だけ。学習ループ（フェーズ5より後）を作る段階で、必要になったら `save-checkpoint` の外の小さな関数として足す。
- 補足: OS のクラッシュ・電源断への耐性は #263 で足した（下の 6）。世代管理とは別の話。

## 6. fsync の費用（#263）

`write-safetensors`（`save-checkpoint`）と vmfb のディスクキャッシュは `with-atomic-output-file`（`src/atomic-file.lisp`）で書く。一時ファイルに書く → `finish-output` → 一時ファイルを `fsync` → rename → 親ディレクトリを `fsync`（この呼び出しで作ったディレクトリがあれば、それぞれの親も）。#263 より前は `finish-output` までで、プロセスが落ちても前のファイルは壊れないが、OS のクラッシュ・電源断の後には rename 済みなのに空か壊れたファイルが残りえた。

macOS の `fsync` はデータをドライブに渡すだけで、ドライブの書き込みキャッシュまでは書き出さない。電源断まで守るには `fcntl(F_FULLFSYNC)` が要り、`nabla:*full-fsync*` を真にすると使う（既定 `nil`、macOS 以外では効果なし。Linux の `fsync` はドライブのキャッシュの書き出しも要求する）。

測り方: 1 つの f32 のテンソルのファイルを同じパスに `write-safetensors` で繰り返し保存し、1 回あたりの実時間を測った（4 KiB は 200 回、1 MiB は 100 回、64 MiB は 10 回の平均。1 回目の書き込みは除く）。「前」は #263 より前の `write-safetensors` を同じイメージで定義し直したもの。環境は Apple M1 / APFS（内蔵 SSD）/ SBCL 2.6.9 / macOS（2026-10-24）。

| ファイルの大きさ | 前（fsync なし） | 後（fsync） | 後（`*full-fsync*` = t） |
| --- | --- | --- | --- |
| 4 KiB | 0.57 ms | 0.55 ms | 8.8 ms |
| 1 MiB | 1.4 ms | 1.8 ms | 10.7 ms |
| 64 MiB | 52 ms | 50 ms | 88 ms |

- macOS の `fsync` は差が測定の揺れに埋もれる（ドライブのキャッシュに渡すだけなので）。学習中に何度も保存しても費用は増えない
- `F_FULLFSYNC` は 1 回あたり約 8 ms 増える（ファイルとディレクトリの 2 回の書き出し待ち）。毎ステップ保存するのでなければ無視できる大きさだが、vmfb のキャッシュの書き込みにも効くので既定は off にした
- Linux（CI、ext4）では `fsync` が実際にディスクへの書き出しを待つので、ここより遅くなりうる。CI では測っていない
