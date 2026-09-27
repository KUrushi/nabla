;;;; IREE のワーカースレッド・コンパイラ呼び出しから見える浮動小数点例外
;;;; トラップ（MXCSR）を、常にすべてマスクしておくための共通マクロ
;;;; （issue #53）。
;;;;
;;;; 背景: SBCL は既定で :overflow :invalid :divide-by-zero の3つのトラップを
;;;; 有効にしている（docs/glossary.md の「float trap」参照）。make-device
;;;; （runtime.lisp）と invoke（execute.lisp）はすでにこの3つを
;;;; sb-int:with-float-traps-masked で包んでいたが、compile-stablehlo 側は
;;;; 元々どのトラップもマスクしていなかった。
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
;;;; 効かない。この理由で make-device・make-session・session-append-module・
;;;; iree-instance の初回生成・invoke を、生成/呼び出しの瞬間にこのマクロで
;;;; 包む（多重防御。iree-instance / make-session がドライバ実装によっては
;;;; 遅延でスレッドを作ることもありうるため、生成されうる全ての箇所を
;;;; 一貫してマスクしておく）。
;;;;
;;;; 実験（このコミットの作業ログ、scratchpad/exp1〜exp3.lisp。fresh な
;;;; sbcl --non-interactive で再現/未再現を確認したもの。以後の版の IREE や
;;;; マシンでは変わりうるので、疑わしければ同じ手順で再確認すること）:
;;;;   - shape (2)・(4 8) の NaN 入力を stablehlo.maximum /
;;;;     compare+select に通す分には、修正前（make-device / invoke の
;;;;     3トラップマスクのみ）でもクラッシュを再現できなかった
;;;;     （:local-task の worker スレッドは make-device の時点ですでに
;;;;     3トラップぶんマスクされているため）。ここでのマスク拡張
;;;;     （5トラップ全部）は、この環境ではまだクラッシュしていない残り2つ
;;;;     （:underflow :inexact）を先回りして塞ぐ多重防御。
;;;;   - ゼロサイズの contracting 次元を持つ dot_general
;;;;     （tensor<2x0xf32> x tensor<0x3xf32>）の backend-compile は、修正前は
;;;;     常に呼び出し元スレッド（コンパイルを実行している Lisp スレッド
;;;;     そのもの）で DIVISION-BY-ZERO を signal した。これは
;;;;     WITH-ALL-FLOAT-TRAPS-MASKED でも glibc の fedisableexcept(3) の
;;;;     直接呼び出しでも再現し続けた（scratchpad/exp2c〜exp2f.lisp。
;;;;     コンパイラのフラグを変えても再現する）ため、SSE の浮動小数点例外
;;;;     （MXCSR、本マクロが制御する対象）ではなく、x86 の整数除算命令
;;;;     （idiv 系、#DE 例外）による 0 除算だと分かった——整数の0除算には
;;;;     マスクビットが存在せず、ソフトウェアでは防げない。SBCL の
;;;;     SIGFPE ハンドラは #DE と浮動小数点例外の両方を DIVISION-BY-ZERO の
;;;;     Lisp コンディションに変換するため、見た目は同じ条件で報告される。
;;;;     プロセス自体は落ちない（SIGFPE はきちんと Lisp コンディションに
;;;;     変換され、非局所脱出で戻れる）ので、compiler.lisp の
;;;;     %compile-stablehlo 側で ARITHMETIC-ERROR を捕まえて
;;;;     IREE-COMPILE-ERROR（:phase :compile）に変換する（このファイルの
;;;;     マスクとは別の対応。IREE/LLVM 側のバグなので、正しい vmfb を
;;;;     得られるようにする根本修正はできない。follow-up 課題）。
;;;;
;;;; マスクするトラップの種類: SBCL 2.2.9.debian（x86-64）が
;;; SB-INT:WITH-FLOAT-TRAPS-MASKED で制御できる5種類すべて
;;; （:underflow :overflow :inexact :invalid :divide-by-zero）。
;;;
;;; ガイダンスにある6種類目の :denormalized-operand は、SBCL のソース
;;; （sb-vm::+float-trap-alist+、src/code/float-trap.lisp）を確認すると
;;; `#+x86` （32bit x86。x87 の FPU 制御ワードにある denormal-operand
;;; トラップビット用）としてだけ定義されており、nabla が対象にする
;;; x86-64（`#+x86-64` は同じ alist に登場しない）では未知のキーワード
;;; として弾かれる（`(sb-int:with-float-traps-masked (:denormalized-operand)
;;; ...)` はマクロ展開時に "unknown float trap kind" のコンパイルエラーに
;;; なる。実際にこのファイルの最初の版で踏んだ）。x86-64 の SSE
;;; （MXCSR）にはそもそも「非正規化オペランド」用の独立した例外マスクビットが
;;; 無く、非正規化数の扱いは denormals-are-zero (DAZ) / flush-to-zero (FTZ)
;;; という別のモードビットで制御する（IREE のカーネル側が
;;; iree_fpu_state_push 相当でこれを設定する。Lisp のトラップとしては
;;; 現れない）。そのため5種類のマスクで、nabla が対象にする環境
;;; （x86-64 Linux）における「SBCL が signal しうる浮動小数点例外」を
;;; すべて覆える。

(in-package #:nabla.iree)

(defmacro with-all-float-traps-masked (&body body)
  "BODY を、SBCL（x86-64）が制御できる5種類の浮動小数点例外トラップすべて
（:underflow :overflow :inexact :invalid :divide-by-zero）をマスクした状態で
評価する。

IREE のワーカースレッド生成点（make-device・make-session・
session-append-module・iree-instance の初回生成）と、LLVM を呼ぶ
コンパイラのエントリポイント（compile-stablehlo・%warm-up-compiler・
ensure-compiler-loaded の LLVM シグナルハンドラ登録点）は、必ずこのマクロで
本体を包むこと（ファイル冒頭のコメント参照。新しいスレッドは生成元スレッドの
MXCSR をそのまま引き継ぐので、生成元スレッドを生成の瞬間にマスクしておく
必要がある）。"
  `(sb-int:with-float-traps-masked
       (:underflow :overflow :inexact :invalid :divide-by-zero)
     ,@body))
