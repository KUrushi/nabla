# embedding の引き: one-hot + dot と gather の比較（issue #240）

`embedding` の引きは、#204 では one-hot と `dot` で書いていた（ids を `[ids... V]` の one-hot にして、表 `[V features]` と `dot` する）。#240 でコアの `gather` に替えた。この文書は、語彙数 V が大きいときの速度とメモリを比べた結果をまとめる。

## 結論

- one-hot 版の計算量とメモリは ids 数 × V × features に比例する（one-hot の配列 `[ids... V]` と、`[ids V] × [V features]` の `dot`）。gather 版は ids 数 × features で、V に依らない
- 勾配も同じ形になる。one-hot 版は one-hot の転置との `dot`、gather 版は scatter-add（大きさ `[V features]` の 0 の配列に、ids 数 × features 個を足し込む）。gather 版の勾配で V に比例するのは、表と同じ形の出力を作る部分だけ
- 下の eager の測定では、V = 50000 のとき one-hot 版は順伝播に約 45 秒・49 GB、勾配に約 92 秒・109 GB を確保した。gather 版は順伝播 0.02 ms、勾配 2.8 ms・7.8 MB だった
- jit（IREE、Colab の VM。下の「jit（IREE）」）では、gather 版の順伝播は local・CUDA とも V に依らず 1 ms 未満だった。勾配は local（CPU）では V = 50000 で gather 版 8 ms・one-hot 版 42 ms と gather 版が速いが、CUDA では gather 版 16 ms・one-hot 版 6 ms と **gather 版（scatter-add）のほうが遅かった**
- 振る舞いの違いが1つある。one-hot 版は表のほかの行にある inf / NaN が引いた行に漏れていた（0 × inf = NaN なので、表のどこかに inf があると、すべての引きのその列が NaN になる）。gather 版は引いた行だけを写すので漏れない（Flax の `jnp.take` と同じ）。範囲外の id が 0 の行になるのは同じ

## 測定

`scripts/bench-embedding.sh`（本体は `scripts/bench-embedding.lisp`）で測った。設定は ids `[8 32]`（256 個）、features 32、f32。V は 1000 / 10000 / 50000。比べたのは、順伝播（引き）と、表についての勾配（損失 `sum(lookup(table, ids) * weights)` の `grad`）の 2 つ。one-hot 版は #204 の `%embedding-apply` を、:i32 の ids と f32 の表に特化してスクリプトの中に書き直したもの。

- 時間: eager（SBCL の上の CPU 実装。jit しない）の呼び出しの中央値。one-hot 版は遅すぎるので 1 回だけ、gather 版は 20 回測った
- メモリ: 1 回の呼び出しで確保したバイト数（`sb-ext:get-bytes-consed` の差）。eager の中間配列の大きさの目安になる。ただしこのカウンタは割り当て領域の単位でしか進まないので、小さな確保（gather 版の順伝播の 32 KiB の出力など）は 0 になることがある。Lisp のヒープの分しか数えないので、jit（IREE）の行には出さない
- 測定環境: SBCL 2.6.9、macOS（Darwin）arm64

| V | 方式 | 順伝播 (ms) | 順伝播で確保 (bytes) | 勾配 (ms) | 勾配で確保 (bytes) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | one-hot | 926.0 | 984,818,240 | 1831.8 | 2,181,018,896 |
| 1000 | gather | 0.021 | 0 | 2.064 | 1,450,976 |
| 10000 | one-hot | 9098.5 | 9,820,709,568 | 18388.5 | 21,753,862,480 |
| 10000 | gather | 0.021 | 0 | 2.192 | 2,698,064 |
| 50000 | one-hot | 45446.9 | 49,102,629,504 | 92060.4 | 108,762,563,216 |
| 50000 | gather | 0.020 | 0 | 2.818 | 7,818,064 |

one-hot 版は時間も確保量も V にほぼ比例して増えた（V が 10 倍で約 10 倍）。eager の `dot` と要素ごとの演算は汎用の Lisp の実装なので、絶対値はコンパイルした実行系よりずっと大きい。それでも比例の関係（ids 数 × V × features）は、jit した StableHLO でも同じになる。one-hot 版の StableHLO は `[256 V]` の比較と convert、`[256 V] × [V 32]` の `dot_general`（256·V·32 回の積和）を持つ。gather 版は `stablehlo.gather` で 256 行（256·32 要素）を写すだけになる。

gather 版の勾配で確保量が V とともに増えるのは、勾配そのもの（表と同じ `[V 32]` の f32、V = 50000 で 6.4 MB）を作るため。順伝播が V に依らないことは、上の数値よりも graph の構造（`gather` の eqn が1つだけ。`tests/nn/layers-test.lisp` の `layers/embedding-lookup-does-not-scale-with-the-vocabulary` が守る）から言える。

## jit（IREE）

`scripts/bench-embedding.sh` の jit の行を、Colab の VM（Tesla T4、Xeon 2 コア、x86_64、SBCL 2.2.9）で local（CPU、local-task）と cuda の両方について測った（2026-10-24）。この PR のスクリプトは local だけを測るので、target を cuda にも替える一時的な変更を加えて測った（その変更は入れていない）。設定は eager と同じ（ids 256 個・features 32・f32）。時間はコンパイル後の呼び出しの中央値（20 回）で、戻り値の to-host を含む。表と ids は先に device array にしてある。**この SBCL の時計の分解能は約 1 ms** で、「<1」は 0.5 ms 未満を表す。

| V | 方式 | local 順伝播 (ms) | local 勾配 (ms) | cuda 順伝播 (ms) | cuda 勾配 (ms) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | one-hot | 1 | 1 | <1 | 1 |
| 1000 | gather | <1 | <1 | <1 | 1 |
| 10000 | one-hot | 6 | 7 | 4 | 2 |
| 10000 | gather | <1 | 2 | <1 | 7 |
| 50000 | one-hot | 39 | 42 | 16 | 6 |
| 50000 | gather | <1 | 8 | <1 | 16 |

- 順伝播: gather 版は local・cuda とも V に依らず 1 ms 未満。one-hot 版は V とともに増え、V = 50000 で local 39 ms・cuda 16 ms
- 勾配（local）: gather 版も V とともに増える（V = 50000 で 8 ms）。勾配の出力 `[V 32]` を 0 で埋めて to-host する分が V に比例するため。それでも one-hot 版の約 1/5
- 勾配（cuda）: gather 版が one-hot 版より遅い（V = 10000 で 7 ms 対 2 ms、V = 50000 で 16 ms 対 6 ms）。どちらも同じ大きさ（`[V 32]`）の勾配を to-host するので、差は scatter-add そのものの費用と見ている。IREE 3.11 の CUDA が `stablehlo.scatter` を並列化しきれていない（重複しうる添字への足し込みを逐次に近い形で下ろしている）可能性があるが、原因は調べていない。CUDA で語彙の大きなモデルを学習するときは、勾配の費用が one-hot 版より増えうる

## 未測定のもの

- 時計の分解能（約 1 ms）より細かい差。測るなら呼び出しを繰り返して平均する
- CUDA の scatter-add が遅い原因（IREE が出すカーネルの形）
- one-hot 版の jit のメモリ（IREE の中の確保。Lisp のヒープの確保量では数えられない）
