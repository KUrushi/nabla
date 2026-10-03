;;;; ad/rules-batch-rng: 乱数まわりのプリミティブのバッチ化ルール（issue #136）。
;;;;
;;;; - shift-right-logical / bitwise-or: 要素演算なので共通ルール %BATCH-ELEMENTWISE。
;;;; - bitcast-convert: バッチ軸を先頭へ動かして適用する。幅が違う bitcast は末尾の次元を
;;;;   増減するので、バッチ軸が末尾にあると意味が変わってしまうため、常に軸 0 にする。
;;;; - rng-bit-generator（複数出力。契約 C3: 出力・軸ともリスト）: 状態のバッチ軸を先頭へ
;;;;   動かし、バッチ次元つきの状態としてそのまま適用する。プリミティブは各行を単独の
;;;;   状態として扱うので（src/primitives/rng.lisp）、出力の各要素は、その要素の状態で
;;;;   単独に呼んだ結果とビット単位で一致する。新しい状態もビットも軸 0。

(in-package #:nabla)

(def-batch-rule shift-right-logical (args batch-dims)
  (%batch-elementwise :shift-right-logical args batch-dims nil))
(def-batch-rule bitwise-or (args batch-dims)
  (%batch-elementwise :bitwise-or args batch-dims nil))

(def-batch-rule bitcast-convert (args batch-dims &key dtype)
  (let ((moved (%vmap-move-axis (first args) (first batch-dims) 0)))
    (values (list (%trace-eqn :bitcast-convert (list moved) :dtype dtype)) (list 0))))

(def-batch-rule rng-bit-generator (args batch-dims &key shape dtype)
  (let ((state (%vmap-move-axis (first args) (first batch-dims) 0)))
    (values (%trace-eqn* :rng-bit-generator (list state) :shape shape :dtype dtype)
            (list 0 0))))
