;;;; README の使用例（examples/add.lisp）が壊れていないことを確かめる
;;;; テスト（issue #15、ビヨンセ・ルール）。examples/add.lisp は README の
;;;; コードブロックとバイト一致させているファイルで、backend プロトコル
;;;; （find-backend :iree → backend-compile → backend-load → backend-invoke
;;;; → to-host）の最小例。ここでは *standard-output* を文字列ストリームに
;;;; 束縛して examples/add.lisp を load し、期待する数値が出力に含まれる
;;;; ことを確認する。純粋な FFI オーケストレーションの疎通確認であり、
;;;; プリミティブや変換のルールではないので mutation testing の対象外
;;;; （nabla-testing スキルの「CFFI の生バインディングの疎通確認は例ベースで
;;;; よい」という考え方をそのまま当てはめている）。

(in-package #:nabla.iree.tests)

(define-iree-test example/add-lisp/prints-expected-sum
    "examples/add.lisp（README の使用例）を読み込むと、標準出力に
4要素の加算結果 \"11.0 22.0 33.0 44.0\" が（順不同ではなく、この並びの
まま）含まれる。README の例が壊れたらこのテストが落ちる。"
  (skip-unless-iree :library :both)
  (let ((output (make-string-output-stream)))
    (let ((*standard-output* output))
      (load (asdf:system-relative-pathname "nabla" "examples/add.lisp")))
    (let ((text (get-output-stream-string output)))
      (is (search "11.0 22.0 33.0 44.0" text)
          "examples/add.lisp の出力に期待する和が見つからなかった: ~S" text))))
