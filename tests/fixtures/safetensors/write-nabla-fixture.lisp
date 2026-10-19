;;;; nabla-written.safetensors を作り直す（issue #198）。リポジトリのルートから:
;;;;
;;;;   export CL_SOURCE_REGISTRY="$PWD/:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:"
;;;;   sbcl --non-interactive --load tests/fixtures/safetensors/write-nabla-fixture.lisp
;;;;
;;;; 作り直したら generate.py で Python がこのファイルを読めることを確かめてコミットする。
;;;; 内容は generate.py の PYTHON_TENSORS と同じ（tests/nn/safetensors-test.lisp の
;;;; fixture-entries）。

(require :asdf)
(asdf:load-system "nabla/nn/tests")

(with-open-file (out "tests/fixtures/safetensors/nabla-written.safetensors"
                     :direction :output :element-type '(unsigned-byte 8) :if-exists :supersede)
  (write-sequence (nabla.nn:encode-safetensors (nabla.nn.tests::fixture-entries)
                                               :metadata '(("format" . "pt") ("note" . "fixture")))
                  out))
