;;;; 外部ライブラリのワーカースレッド・コンパイラ呼び出しから見える
;;;; 浮動小数点例外トラップ（MXCSR）を、常にすべてマスクしておくための
;;;; 共通マクロ（issue #53、#79 で FFI 連携の層から移動）。
;;;;
;;;; 背景: SBCL は既定で :overflow :invalid :divide-by-zero の3つのトラップを
;;;; 有効にしている（docs/glossary.md の「float trap」参照）。
;;;;
;;;; Linux はスレッド生成（clone(2)）時に、生成元スレッドの MXCSR
;;;; （SSE の浮動小数点制御・ステータスレジスタ、トラップマスクを含む）を
;;;; そのまま新しいスレッドへコピーする。sb-int:with-float-traps-masked は
;;;; 「呼び出したスレッドの」MXCSR だけを書き換えて、動的エクステントを
;;;; 抜けるときに元に戻す（thread-local な設定で、プロセス全体には効かない）。
;;;; そのため、あるスレッドを生成する瞬間に「生成元スレッドの」MXCSR が
;;;; マスクされていないと、生成された新しいスレッドは未マスクな MXCSR を
;;;; 引き継いでしまい、以後 with-float-traps-masked をいくら重ねても
;;;; （呼び出し元スレッドの MXCSR しか変えないので）その新しいスレッドには
;;;; 効かない。外部ライブラリがワーカースレッドを作りうる呼び出し（生成の
;;;; 瞬間）と、コンパイル中に内部のスレッドを作るライブラリ呼び出しを、
;;;; このマクロで包む。
;;;;
;;;; 実験の記録（再現手順・経路の詳細）は docs/float-traps-experiments.md
;;;; にある。
;;;;
;;;; マスクするトラップの種類: SBCL 2.2.9.debian（x86-64）が
;;;; SB-INT:WITH-FLOAT-TRAPS-MASKED で制御できる5種類すべて
;;;; （:underflow :overflow :inexact :invalid :divide-by-zero）。
;;;; 6種類目の :denormalized-operand は SBCL では 32bit x86（x87）専用で、
;;;; x86-64 ではマクロ展開時に "unknown float trap kind" になる。x86-64 の
;;;; SSE には非正規化オペランド用の独立した例外マスクビットが無く、
;;;; 非正規化数の扱いは DAZ / FTZ という別のモードビットで決まる
;;;; （Lisp のトラップとしては現れない）。

(in-package #:nabla.ffi-support)

(defmacro with-all-float-traps-masked (&body body)
  "BODY を、SBCL（x86-64）が制御できる5種類の浮動小数点例外トラップすべて
（:underflow :overflow :inexact :invalid :divide-by-zero）をマスクした状態で
評価する。

外部ライブラリがワーカースレッドを生成しうる呼び出しや、内部で
スレッドを作るコンパイラ呼び出しは、必ずこのマクロで本体を包むこと
（ファイル冒頭のコメント参照。新しいスレッドは生成元スレッドの MXCSR を
そのまま引き継ぐので、生成元スレッドを生成の瞬間にマスクしておく必要がある）。

WITH-LISP-SIGNAL-HANDLERS-PRESERVED（signals.lisp）と併用するときの入れ子の
順序はどちらでもよい。MXCSR のマスク（スレッドローカルな浮動小数点例外の
設定）と、sigaction によるプロセス全体のシグナルハンドラの保存・復元は
互いに独立したオペレーティングシステムの状態だから。"
  `(sb-int:with-float-traps-masked
       (:underflow :overflow :inexact :invalid :divide-by-zero)
     ,@body))
