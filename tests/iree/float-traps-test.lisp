;;;; 浮動小数点例外トラップのマスク（issue #53）の medium テスト。
;;;;
;;;; 3つの性質を確かめる:
;;;;   1. NaN を含む入力を compare（6方向）+ select・maximum・minimum に通す
;;;;      と、IREE の local backend での実行結果は eager 実装と NaN の位置が
;;;;      一致し、それ以外の要素は許容誤差つきで一致する（プロセスが落ちない
;;;;      こと自体は、このテストが最後まで走って fiveam の結果を報告できる
;;;;      ことで保証される）。ALLCLOSE は NaN を含む要素をすべて不一致と
;;;;      みなす（tests/support/allclose.lisp）ので、ここでは NaN の位置を
;;;;      別にチェックする %NAN-AWARE-MATCH-P を使う。ただし、このケース
;;;;      （NaN を実行時の引数として渡す）は、issue #53 の実際のクラッシュを
;;;;      再現しない（:local-task の worker スレッドは make-device の時点で
;;;;      すでにマスク済みのため）。実際にクラッシュするのは次の 1b。
;;;;   1b. stablehlo.constant に NaN を埋め込んだモジュール（引数なし）を
;;;;      BACKEND-COMPILE すると、IREE のコンパイル時定数畳み込み
;;;;      （JitGlobalsPass）がコンパイル中の Lisp スレッド上で NaN の
;;;;      compare+select+maximum を JIT 実行する。この JIT のワーカー
;;;;      スレッドはコンパイルスレッドから生成されるためその MXCSR を
;;;;      引き継ぎ、compile-stablehlo 自身をマスクしていないとここで
;;;;      SIGFPE によりプロセスごと落ちる——これが issue #53 の実際の
;;;;      トリガーで、compile-stablehlo へのマスク追加が本 PR の
;;;;      load-bearing な修正そのもの（docs/float-traps-experiments.md
;;;;      参照）。
;;;;   2. ゼロサイズの contracting 次元を持つ dot_general の COMPILE-STABLEHLO
;;;;      は、生の DIVISION-BY-ZERO（Lisp コンディション）を漏らさない。
;;;;      docs/float-traps-experiments.mdのとおり、この特定のケースは
;;;;      x86 の整数 0 除算（#DE）が原因で、浮動小数点トラップのマスクでは
;;;;      防げない既知の IREE/LLVM 側の制約なので、compiler.lisp 側で
;;;;      IREE-COMPILE-ERROR に変換している。
;;;;
;;;;      issue #68: この #DE は ireeCompilerInvocationPipeline が C++
;;;;      オブジェクト（mlir::OpPassManager 等）を構築している最中に起き、
;;;;      SBCL の非局所脱出はその構築中の C++ フレームのデストラクタを
;;;;      一切走らせずに飛び越える。実験で確認したとおり（compiler.lisp の
;;;;      *compiler-poison-reason* 冒頭のコメント参照）、この #DE 単体は
;;;;      その場ではプロセスを壊さないが、その後さらに多数の distinct な
;;;;      コンパイルを重ねると in-process の libIREECompiler.so が
;;;;      SB-SYS:MEMORY-FAULT-ERROR で壊れうる。そのため compiler.lisp は
;;;;      この #DE を観測したプロセスを以後 poisoned として扱い、次回以降の
;;;;      compile-stablehlo を :phase :poisoned の IREE-COMPILE-ERROR で
;;;;      即座に失敗させる。このファイルの他のテストや :nabla.medium 全体と
;;;;      同じ共有プロセスで実際にこの #DE を起こすと、以後そのプロセスでの
;;;;      コンパイルが軒並み :phase :poisoned で失敗してしまうため、この
;;;;      性質のテスト（float-traps/zero-size-dot-general-poisons-compiler-then-fails-clearly）
;;;;      だけは真っさらな子 SBCL プロセスの中で #DE を起こし、
;;;;      (1) 生の DIVISION-BY-ZERO が漏れないこと、(2) その直後の別の
;;;;      compile-stablehlo が :phase :poisoned で明確に失敗することを
;;;;      確かめる。実際のトレース経路（with-tracing → trace-to-graph →
;;;;      emit-stablehlo）での K=0 の dot-general は、issue #62 の修正で
;;;;      dot-general の :emit（src/primitives/dot.lisp）がこの形の
;;;;      dot_general をもう出さなくなった（ゼロ定数を出す）ため、この #DE
;;;;      をそもそも踏まない。その経路のコンパイル・実行の確認は
;;;;      tests/iree/dot-test.lisp の
;;;;      dot-general/zero-contracting-compiles-and-matches-eager で行う。
;;;;
;;;; 3つ目の性質として、繰り返し make-device / invoke しても、呼び出した
;;;; スレッド自身の浮動小数点トラップの設定が変わらないこと
;;;; （sb-int:get-floating-point-modes が呼び出し前後で等しいこと）も
;;;; 確かめる（with-float-traps-masked は動的エクステントを抜けるときに
;;;; 必ず元に戻すはずで、これが崩れていないかの回帰テスト）。

(in-package #:nabla.iree.tests)

(defparameter *float-traps-nan-f32*
  (sb-kernel:make-single-float #x7fc00000)
  "quiet NaN の single-float（ビットパターン 0x7fc00000）。")

(defun %float-traps-nan-array (shape nan-positions)
  "SHAPE（次元のリスト）の single-float 配列を作る。行優先の添字が
NAN-POSITIONS（整数のリスト）に含まれる要素は NaN、それ以外は
（添字に応じて変化する）通常の値にする。"
  (let ((result (make-array shape :element-type 'single-float)))
    (dotimes (i (array-total-size result) result)
      (setf (row-major-aref result i)
            (if (member i nan-positions)
                *float-traps-nan-f32*
                (coerce (- (mod i 7) 3) 'single-float))))))

(defun %nan-aware-match-p (actual expected dtype)
  "ACTUAL と EXPECTED（DTYPE の格納表現を持つ配列）が、要素ごとに
「両方 NaN」または「許容誤差つきで一致（APPROX=）」のどちらかを満たすか。
ALLCLOSE と違い、NaN を許容不一致にしない（NaN の伝播そのものを確かめる
テストのため）。形状が違えば NIL を返す。"
  (unless (equal (array-dimensions actual) (array-dimensions expected))
    (return-from %nan-aware-match-p nil))
  (let ((decoded-actual (decode-array actual dtype))
        (decoded-expected (decode-array expected dtype)))
    (dotimes (i (array-total-size decoded-actual) t)
      (let ((a (row-major-aref decoded-actual i))
            (e (row-major-aref decoded-expected i)))
        (unless (if (sb-ext:float-nan-p e)
                    (sb-ext:float-nan-p a)
                    (approx= a e :dtype dtype))
          (return-from %nan-aware-match-p nil))))))

;;; --- 性質1: NaN を通す compare+select / maximum / minimum ---

(define-iree-test float-traps/nan-compare-select-matches-eager
    "NaN を含む f32 入力を select(compare(a, b, direction), a, b) に通した
IREE の実行結果は、6方向すべてで eager 実装と NaN の位置・値が一致する
（issue #53）。"
  (skip-unless-iree :library :both)
  (let* ((aval (nb:make-aval '(4 8) :f32))
         (pred-aval (nb:make-aval '(4 8) :i1))
         (a (%float-traps-nan-array '(4 8) '(0 5 17 31)))
         (b (%float-traps-nan-array '(4 8) '(1 5 20))))
    (dolist (direction '(:lt :le :gt :ge :eq :ne))
      (let ((body (list (format nil "%c = stablehlo.compare ~A, %a0, %a1 : (~A, ~A) -> ~A"
                                 (symbol-name direction)
                                 (nb::tensor-type-string aval) (nb::tensor-type-string aval)
                                 (nb::tensor-type-string pred-aval))
                         (format nil "%0 = stablehlo.select %c, %a0, %a1 : ~A, ~A"
                                 (nb::tensor-type-string pred-aval) (nb::tensor-type-string aval)))))
        (with-one-op-module ((backend module) (list aval aval) aval body)
          (let* ((pred (funcall (nb::primitive-eager (nb::find-primitive :compare))
                                 (list a b) (list aval aval) :direction direction))
                 (expected (funcall (nb::primitive-eager (nb::find-primitive :select))
                                     (list pred a b) (list pred-aval aval aval))))
            (with-device-arrays ((da (to-device a backend :dtype :f32))
                                 (db (to-device b backend :dtype :f32)))
              (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                (is (%nan-aware-match-p (to-host result) expected :f32)
                    "compare(~A)+select: NaN を含む IREE の実行結果が eager 実装と一致しなかった"
                    direction)))))))))

(defmacro def-float-traps-minmax-test (test-name prim-name mlir-op)
  "PRIM-NAME（:MAX/:MIN）・MLIR-OP から、NaN を含む f32 入力に対する
stablehlo.MLIR-OP の IREE 実行結果が eager 実装と NaN の位置・値が一致する
ことを確かめる DEFINE-IREE-TEST を作る。"
  `(define-iree-test ,test-name
       ,(format nil "NaN を含む f32 入力を stablehlo.~A に通した IREE の実行結果は、
eager ~(~A~) 実装と NaN の位置・値が一致する（issue #53）。" mlir-op prim-name)
     (skip-unless-iree :library :both)
     (let* ((aval (nb:make-aval '(4 8) :f32))
            (a (%float-traps-nan-array '(4 8) '(0 5 17 31)))
            (b (%float-traps-nan-array '(4 8) '(1 5 20))))
       (with-one-op-module
           ((backend module) (list aval aval) aval
            (list (format nil "%0 = stablehlo.~A %a0, %a1 : ~A" ,mlir-op (nb::tensor-type-string aval))))
         (let ((expected (funcall (nb::primitive-eager (nb::find-primitive ,prim-name))
                                   (list a b) (list aval aval))))
           (with-device-arrays ((da (to-device a backend :dtype :f32))
                                (db (to-device b backend :dtype :f32)))
             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
               (is (%nan-aware-match-p (to-host result) expected :f32)
                   ,(format nil "~(~A~): NaN を含む IREE の実行結果が eager 実装と一致しなかった" prim-name)))))))))

(def-float-traps-minmax-test float-traps/nan-maximum-matches-eager :max "maximum")
(def-float-traps-minmax-test float-traps/nan-minimum-matches-eager :min "minimum")

;;; --- 性質1b: コンパイル時定数畳み込み（JitGlobalsPass）に NaN 定数を通す ---
;;;
;;; 上の性質1（引数として渡す NaN）は、修正前でもクラッシュを再現できな
;;; かった（:local-task の worker スレッドは make-device の時点ですでに
;;; 3トラップぶんマスクされていたため。docs/float-traps-experiments.md
;;; 参照）。実際に issue #53 のクラッシュを起こすのは、NaN を引数ではなく
;;; モジュールの stablehlo.constant に埋め込み、コンパイル時定数畳み込み
;;; （IREE の JitGlobalsPass。実行時ではなくコンパイル中の Lisp スレッド上
;;; で JIT されたコードが走る）に通す経路で、その JIT コードが生成される
;;; スレッドは「コンパイルを実行している Lisp スレッド」から fork/clone
;;; されるため、そのスレッドの MXCSR をそのまま引き継ぐ。
;;; compile-stablehlo 自身を WITH-ALL-FLOAT-TRAPS-MASKED で包んでいないと、
;;; このスレッドが未マスクな状態で NaN の compare/maximum を実行し、
;;; SIGFPE でプロセスごと落ちる（"in non-lisp tid ... resignaling to a
;;; lisp tid" で終了コード136。fresh sbcl + fresh NABLA_CACHE_DIR での再現
;;; 手順は PR の説明参照）。vmfb ディスクキャッシュが当たると
;;; %compile-stablehlo 自体が呼ばれずこの経路を通らないので、
;;; *compile-cache-directory* を NIL に束縛してキャッシュを毎回外す。

(defun %f32-dense-constant-literal (values aval)
  "VALUES（single-float のリスト、行優先）から AVAL 型の
stablehlo.constant 用 dense<...> リテラルを組み立てる。NaN は
0x7FC00000 のような32ビットのビットパターン16進表記、それ以外は10進で書く
（StableHLO の完全なリテラル構文パーサではなく、この2ケースだけ扱う最小の
実装）。"
  (declare (ignore aval))
  (format nil "dense<[~{~A~^, ~}]>"
          (mapcar (lambda (v)
                    (if (sb-ext:float-nan-p v)
                        (format nil "0x~8,'0X" (nb::%single-float-bits v))
                        (format nil "~F" v)))
                  values)))

(define-iree-test float-traps/nan-constant-fold-does-not-crash-compiler
    "モジュールの stablehlo.constant に埋め込んだ NaN を compare + select +
maximum に通した StableHLO を BACKEND-COMPILE すると、IREE のコンパイル時
定数畳み込み（JitGlobalsPass）がこの経路を JIT 実行する。プロセスが落ちず
（このテストが最後まで走って結果を報告できることで保証される）、実行結果は
eager 実装と NaN の位置・値が一致する（issue #53 の実際のクラッシュ経路）。"
  (skip-unless-iree :library :both)
  (let* ((aval (nb:make-aval '(4) :f32))
         (pred-aval (nb:make-aval '(4) :i1))
         (a (make-array 4 :element-type 'single-float
                           :initial-contents (list *float-traps-nan-f32* 1.0 *float-traps-nan-f32* 2.0)))
         (b (make-array 4 :element-type 'single-float
                           :initial-contents (list 1.0 *float-traps-nan-f32* *float-traps-nan-f32* 3.0)))
         (text (format nil "func.func @main() -> ~A {~%  ~
%a = stablehlo.constant ~A : ~A~%  ~
%b = stablehlo.constant ~A : ~A~%  ~
%c = stablehlo.compare LT, %a, %b : (~A, ~A) -> ~A~%  ~
%s = stablehlo.select %c, %a, %b : ~A, ~A~%  ~
%0 = stablehlo.maximum %s, %b : ~A~%  ~
func.return %0 : ~A~%}"
                       (nb::tensor-type-string aval)
                       (%f32-dense-constant-literal (coerce a 'list) aval) (nb::tensor-type-string aval)
                       (%f32-dense-constant-literal (coerce b 'list) aval) (nb::tensor-type-string aval)
                       (nb::tensor-type-string aval) (nb::tensor-type-string aval) (nb::tensor-type-string pred-aval)
                       (nb::tensor-type-string pred-aval) (nb::tensor-type-string aval)
                       (nb::tensor-type-string aval)
                       (nb::tensor-type-string aval)))
         (expected (let* ((pred (funcall (nb::primitive-eager (nb::find-primitive :compare))
                                          (list a b) (list aval aval) :direction :lt))
                          (sel (funcall (nb::primitive-eager (nb::find-primitive :select))
                                        (list pred a b) (list pred-aval aval aval))))
                     (funcall (nb::primitive-eager (nb::find-primitive :max))
                              (list sel b) (list aval aval))))
         (backend (nabla:find-backend :iree))
         (nabla:*compile-cache-directory* nil))
    (let ((module (nabla:backend-load backend (nabla:backend-compile backend text))))
      (unwind-protect
           (with-device-arrays ((result (nabla:backend-invoke backend module "main")))
             (is (%nan-aware-match-p (to-host result) expected :f32)
                 "定数畳み込みを通した NaN compare+select+maximum の IREE 実行結果が eager 実装と一致しなかった"))
        (nabla:backend-unload backend module)))))

;;; --- 性質2: ゼロサイズの contracting 次元を持つ dot_general ---

(defparameter *zero-size-dot-general-poison-check-child-source*
  "(require :asdf)
(asdf:load-system \"nabla/iree\")
(in-package :nabla.iree)

;; issue #68: 実際に SB-SYS:MEMORY-FAULT-ERROR を踏んだ再現手順は、他の
;; distinct なコンパイルを多数（125個）行った *後に* この #DE を起こす、
;; というものだった。ここでも、まず20個の distinct な StableHLO を
;; コンパイルしてから #DE を起こす（少ない回数でも安全側に倒すため
;; 125 ではなく20にしているが、この後の poison チェックは
;; compiler.lisp が #DE を観測した時点で無条件に効くので、実際には
;; 事前のコンパイル回数に依存しない——それ自体も、この回帰テストが
;; 保証する契約の一部）。
(dotimes (n 20)
  (let ((text (format nil \"func.func @main(%a0: tensor<~Dx~Dxf32>) -> tensor<~Dx~Dxf32> {~%  func.return %a0 : tensor<~Dx~Dxf32>~%}\"
                       (1+ n) (1+ n) (1+ n) (1+ n) (1+ n) (1+ n))))
    (unless (plusp (length (compile-stablehlo text)))
      (format t \"PRE-VOLUME-RESULT=FAILED-AT-~D~%\" n)
      (sb-ext:exit :code 1))))
(format t \"PRE-VOLUME-RESULT=OK~%\")

(defparameter *k0-text* \"func.func @main(%a0: tensor<2x0xf32>, %a1: tensor<0x3xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.dot_general %a0, %a1, contracting_dims = [1] x [0] : (tensor<2x0xf32>, tensor<0x3xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}\")
(defparameter *trivial-text* \"func.func @main() { return }\")

(handler-case
    (progn (compile-stablehlo *k0-text*)
           (format t \"K0-RESULT=COMPILED-OK~%\"))
  (iree-compile-error (c)
    (format t \"K0-RESULT=COMPILE-ERROR PHASE=~A~%\" (iree-compile-error-phase c)))
  (division-by-zero ()
    (format t \"K0-RESULT=RAW-DIVISION-BY-ZERO-LEAKED~%\")))

(handler-case
    (progn (compile-stablehlo *trivial-text*)
           (format t \"FOLLOWUP-RESULT=COMPILED-OK~%\"))
  (iree-compile-error (c)
    (format t \"FOLLOWUP-RESULT=COMPILE-ERROR PHASE=~A~%\" (iree-compile-error-phase c))))

(sb-ext:exit :code 0)
"
  "%RUN-IN-CHILD-SBCL に渡す、子プロセス側の完全なソース。support.lisp の
%RUN-IN-CHILD-SBCL を使う（issue #68。ファイル冒頭のコメント参照）。")

(define-iree-test float-traps/zero-size-dot-general-poisons-compiler-then-fails-clearly
    "多数の distinct な compile-stablehlo（20個）に続けて、ゼロサイズの
contracting 次元を持つ dot_general
（tensor<2x0xf32> x tensor<0x3xf32> -> tensor<2x3xf32>）の生 MLIR を
compile-stablehlo に通すと、x86 の整数0除算（#DE、マスクできない。
docs/float-traps-experiments.md参照）が起き、生の DIVISION-BY-ZERO
（Lisp コンディション）は漏れず、IREE-COMPILE-ERROR として報告される
（issue #53）。

issue #68: この #DE の直後に行う別の（問題のない）compile-stablehlo は、
:phase :poisoned の IREE-COMPILE-ERROR で明確に失敗し、メモリ不正アクセスへ
進まない（ファイル冒頭のコメント、compiler.lisp の *compiler-poison-reason*
冒頭のコメント参照）。#DE を起こす経路は、このファイルの他のテストや
:nabla.medium 全体と同じ共有プロセスを汚染しないよう、常に真っさらな子
SBCL プロセス（%RUN-IN-CHILD-SBCL、tests/iree/support.lisp）の中で実行する。"
  (skip-unless-iree :library :compiler)
  (multiple-value-bind (exit-code output error-output)
      (%run-in-child-sbcl *zero-size-dot-general-poison-check-child-source*)
    (is (= 0 exit-code)
        "child process exited ~D (should always exit 0, even when the #DE path is hit); stdout:~%~A~%stderr:~%~A"
        exit-code output error-output)
    (is (search "PRE-VOLUME-RESULT=OK" output)
        "child did not finish its 20 pre-volume compiles; output:~%~A" output)
    (is (not (search "K0-RESULT=RAW-DIVISION-BY-ZERO-LEAKED" output))
        "child leaked a raw DIVISION-BY-ZERO instead of IREE-COMPILE-ERROR (issue #53 regressed); output:~%~A"
        output)
    (is (or (search "K0-RESULT=COMPILED-OK" output) (search "K0-RESULT=COMPILE-ERROR" output))
        "child did not report a recognized K0 result; output:~%~A" output)
    (if (search "K0-RESULT=COMPILE-ERROR" output)
        (is (search "FOLLOWUP-RESULT=COMPILE-ERROR PHASE=POISONED" output)
            "after the #DE-triggering compile, a later ordinary compile should fail with :phase :poisoned ~
instead of proceeding on corrupted in-process compiler state (issue #68); output:~%~A" output)
        (is (search "FOLLOWUP-RESULT=COMPILED-OK" output)
            "K0 compiled without hitting the #DE, so the compiler should not have been poisoned; output:~%~A"
            output))))

;;; --- 性質3: 呼び出しスレッド自身のトラップ設定は変わらない ---

(define-iree-test float-traps/repeated-invoke-preserves-calling-thread-modes
    "make-device と invoke を繰り返しても、呼び出したスレッド自身の
浮動小数点トラップの設定（sb-int:get-floating-point-modes の :traps）は
呼び出し前後で変わらない（with-all-float-traps-masked が動的エクステントを
抜けるときに必ず元へ戻すことの回帰テスト。issue #53）。"
  (skip-unless-iree :library :both)
  (let* ((aval (nb:make-aval '(4) :f32))
         (before (sb-int:get-floating-point-modes)))
    (with-one-op-module
        ((backend module) (list aval aval) aval
         (list (format nil "%0 = stablehlo.maximum %a0, %a1 : ~A" (nb::tensor-type-string aval))))
      (dotimes (i 10)
        (let ((a (make-array 4 :element-type 'single-float :initial-element (coerce i 'single-float)))
              (b (make-array 4 :element-type 'single-float :initial-element 1.0)))
          (with-device-arrays ((da (to-device a backend :dtype :f32))
                               (db (to-device b backend :dtype :f32)))
            (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
              result)))))
    (is (equal (getf before :traps) (getf (sb-int:get-floating-point-modes) :traps))
        "make-device/invoke を繰り返した後、呼び出しスレッドの float trap 設定が変わっていた")))
