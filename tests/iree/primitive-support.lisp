;;;; ONE-OP-MODULE-TEXT / WITH-ONE-OP-MODULE: 1つの StableHLO 演算だけを
;;;; 含む @main 関数を組み立て、IREE でコンパイル・ロードするヘルパー
;;;; （issue #31 p1）。
;;;;
;;;; add/sub/mul/div 以外の二項・単項プリミティブ（p2/p3）もこのファイルを
;;;; 使う想定（契約 §4「p1 が tests/iree/primitive-support.lisp を持つ」）。
;;;; 1回のコンパイルは約350msかかる（tests/iree/backend-test.lisp のコメント
;;;; 参照）ので、WITH-ONE-OP-MODULE は本体の外でコンパイル・ロードを1回だけ
;;;; 行い、check-it の各試行はロード済みの module を使い回す。

(in-package #:nabla.iree.tests)

(defun one-op-module-text (in-avals out-aval body-lines)
  "IN-AVALS（AVAL のリスト）を %a0, %a1, ... という引数にした @main 関数の
StableHLO テキストを組み立てる。BODY-LINES は %a0.. を参照し、最後に %0 を
定義する行（インデント無し）のリスト。返り値は OUT-AVAL の型で、%0 を返す。

func.func @main(%a0: T0, %a1: T1) -> Tout {
  <body lines>
  func.return %0 : Tout
}"
  (with-output-to-string (out)
    (format out "func.func @main(~{~A~^, ~}) -> ~A {~%"
            (loop for in-aval in in-avals
                  for i from 0
                  collect (format nil "%a~D: ~A" i (nb::tensor-type-string in-aval)))
            (nb::tensor-type-string out-aval))
    (dolist (line body-lines)
      (format out "  ~A~%" line))
    (format out "  func.return %0 : ~A~%}" (nb::tensor-type-string out-aval))))

(defmacro with-one-op-module (((backend module) in-avals out-aval body-lines) &body body)
  "IN-AVALS / OUT-AVAL / BODY-LINES から ONE-OP-MODULE-TEXT を組み立て、
find-backend :iree でコンパイル・ロードして BACKEND / MODULE に束縛し、
BODY を評価してから backend-unload する。"
  `(let* ((,backend (nabla:find-backend :iree))
          (,module (nabla:backend-load
                    ,backend
                    (nabla:backend-compile ,backend (one-op-module-text ,in-avals ,out-aval ,body-lines)))))
     (unwind-protect
          (progn ,@body)
       (nabla:backend-unload ,backend ,module))))
