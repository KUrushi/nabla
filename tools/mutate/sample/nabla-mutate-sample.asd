;;;; sample.asd -- nabla-mutate の runner を確かめるための小さなサンプル
;;;;
;;;; nabla-mutate/tests から使う。それ自体は nabla にも nabla-mutate にも
;;;; 依存しない、完全に自己完結した Lisp コード。

(defsystem "nabla-mutate-sample"
  :description "mutation testing runner を確かめるためのサンプル関数"
  :pathname "src"
  :components ((:file "sample")))

(defsystem "nabla-mutate-sample/weak-tests"
  :description "わざと弱いテスト（エラーが出ないことしか見ない）"
  :depends-on ("nabla-mutate-sample")
  :pathname "weak-tests"
  :components ((:file "weak")))

(defsystem "nabla-mutate-sample/strong-tests"
  :description "値と境界を確かめる強いテスト"
  :depends-on ("nabla-mutate-sample")
  :pathname "strong-tests"
  :components ((:file "strong")))
