;;;; shape-primitive-support: reshape / broadcast-in-dim / transpose の medium
;;;; テストが共有する「1つの op だけを持つモジュール」のビルダー（issue #31
;;;; p4）。
;;;;
;;;; チェーンAの tests/iree/primitive-support.lisp とほぼ同じ形だが、
;;;; チェーンBはそのファイルを編集しない約束（契約 §1）なので、ここに
;;;; 名前を SHAPE- 接頭辞にした自分のコピーを持つ（wave 3 で重複を解消する
;;;; 予定）。

(in-package #:nabla.iree.tests)

(defun shape-one-op-module-text (in-avals out-aval body-lines)
  "IN-AVALS（AVAL のリスト）・OUT-AVAL・BODY-LINES（%a0, %a1, ... を参照し、
最後に %0 を定義する MLIR 行のリスト）から、無名の1関数 @main だけを持つ
StableHLO のテキストを組み立てる。"
  (format nil "func.func @main(~{~A~^, ~}) -> ~A {~%~{  ~A~%~}  func.return %0 : ~A~%}"
          (loop for in-aval in in-avals
                for i from 0
                collect (format nil "%a~D: ~A" i (nb::tensor-type-string in-aval)))
          (nb::tensor-type-string out-aval)
          body-lines
          (nb::tensor-type-string out-aval)))

(defmacro with-shape-one-op-module ((backend module) in-avals out-aval body-lines &body body)
  "(FIND-BACKEND :IREE) を BACKEND に束縛し、SHAPE-ONE-OP-MODULE-TEXT の
結果を BACKEND-COMPILE → BACKEND-LOAD した MODULE を束縛して BODY を
評価する。BODY を抜けたら（非局所脱出でも）BACKEND-UNLOAD する。
コンパイルは（呼び出しごとに）1回だけなので、複数の seed を試す
check-it は BODY の中で回すこと（契約 §4 のテスト点5）。"
  `(let* ((,backend (nabla:find-backend :iree))
          (,module (nabla:backend-load
                    ,backend
                    (nabla:backend-compile ,backend (shape-one-op-module-text ,in-avals ,out-aval ,body-lines)))))
     (unwind-protect
          (progn ,@body)
       (nabla:backend-unload ,backend ,module))))

(defun shape-primitive-eager (name arrays in-avals &rest params)
  "NAME（:RESHAPE / :BROADCAST-IN-DIM / :TRANSPOSE）の eager 実装を、
device 実行の期待値（オラクル）として呼ぶ。"
  (apply (nb::primitive-eager (nb::find-primitive name)) arrays in-avals params))
