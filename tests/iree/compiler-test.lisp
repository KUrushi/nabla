;;;; nabla.iree のコンパイラ埋め込み C API バインディングのテスト。
;;;;
;;;; CFFI バインディング自体は mutation testing の対象外（CLAUDE.md /
;;;; issue #5 の補足）。ここでの主眼は「公開 API を通した疎通」と
;;;; 「壊れた入力でプロセスが落ちないこと」。

(in-package #:nabla.iree.tests)

(defun %locked-iree-commit ()
  "third_party/iree.lock の commit=... の値を返す。"
  (let ((path (asdf:system-relative-pathname "nabla" "third_party/iree.lock")))
    (with-open-file (stream path :direction :input)
      (loop for line = (read-line stream nil nil)
            while line
            when (and (>= (length line) 7) (string= "commit=" line :end2 7))
              return (string-trim '(#\Space #\Tab #\Return) (subseq line 7))))))

(defun %vmrss-kb ()
  "/proc/self/status の VmRSS（キロバイト）。Linux 専用。"
  (with-open-file (stream "/proc/self/status" :direction :input)
    (loop for line = (read-line stream nil nil)
          while line
          when (and (>= (length line) 6) (string= "VmRSS:" line :end2 6))
            return (parse-integer line :start 6 :junk-allowed t))))

(define-iree-test compiler/compile-stablehlo/matmul-fixture-produces-zip-wrapped-vmfb
    "手書きの matmul フィクスチャをコンパイルすると、非空で先頭4バイトが
ZIP local-file-header シグネチャ（IREE の polyglot zip 形式の vmfb）の
バイト列が得られる。"
  (skip-unless-iree :library :compiler)
  (let ((bytes (compile-stablehlo (stablehlo-fixture "matmul"))))
    (is (typep bytes '(simple-array (unsigned-byte 8) (*))))
    (is (plusp (length bytes)))
    (is (equalp *vmfb-magic* (subseq bytes 0 4)))))

(define-iree-test compiler/compile-stablehlo/malformed-text-signals-compile-error
    "MLIR として文法の誤った StableHLO をコンパイルすると、少なくとも1つの
:error 診断を含む iree-compile-error が signal され、プロセスは落ちない。"
  (skip-unless-iree :library :compiler)
  (is (check-it (generator (map (lambda (garbage)
                                   (concatenate 'string "func.func @main() { " garbage))
                                 (string)))
                (lambda (text)
                  (handler-case
                      (progn (compile-stablehlo text) nil)
                    (iree-compile-error (condition)
                      (some (lambda (diagnostic) (eq (car diagnostic) :error))
                            (iree-compile-error-diagnostics condition)))))
                :regression-id compiler/compile-stablehlo/malformed-text-signals-compile-error
                :regression-file (regression-path "iree-compiler-malformed" :package "NABLA.IREE.TESTS")))
  ;; 具体例1: 未知の演算。
  (handler-case
      (progn
        (compile-stablehlo "func.func @main() {
  \"stablehlo.bogus\"() : () -> ()
  func.return
}")
        (fiveam:fail "unknown op ~S should have signalled iree-compile-error" "stablehlo.bogus"))
    (iree-compile-error (condition)
      (is (some (lambda (diagnostic) (search "unregistered operation 'stablehlo.bogus'" (cdr diagnostic)))
                (iree-compile-error-diagnostics condition)))))
  ;; 具体例2: 返り値の型不一致。
  (handler-case
      (progn
        (compile-stablehlo "func.func @main(%a: tensor<2x3xf32>) -> tensor<3x3xf32> {
  func.return %a : tensor<2x3xf32>
}")
        (fiveam:fail "return type mismatch should have signalled iree-compile-error"))
    (iree-compile-error (condition)
      (is (some (lambda (diagnostic) (search "doesn't match function result type" (cdr diagnostic)))
                (iree-compile-error-diagnostics condition))))))

(define-iree-test compiler/compile-stablehlo/repeated-compiles-are-stable
    "同じ StableHLO を繰り返しコンパイルしても、結果のバイト列は毎回同じで、
メモリ使用量（RSS）が際限なく増え続けない。"
  (skip-unless-iree :library :compiler)
  (let* ((text (stablehlo-fixture "matmul"))
         (first (compile-stablehlo text)))
    (dotimes (i 19)
      (is (equalp first (compile-stablehlo text))))
    (let ((rss-before (%vmrss-kb)))
      (dotimes (i 20)
        (compile-stablehlo text))
      (let ((rss-after (%vmrss-kb)))
        (when (and rss-before rss-after)
          (is (< (- rss-after rss-before) (* 64 1024))
              "RSS grew by ~A KB over 20 compiles (before=~A after=~A)"
              (- rss-after rss-before) rss-before rss-after))))))

(fiveam:test (compiler/compile-flags/targets-are-deterministic :suite :nabla.small)
  "compile-flags は同じ TARGET に対して毎回同じフラグのリストを返し、
:local の先頭は StableHLO 入力を指定するフラグで、未知の TARGET はエラーになる。"
  (is (equal (compile-flags :local) (compile-flags :local)))
  (is (string= "--iree-input-type=stablehlo" (first (compile-flags :local))))
  (is (equal (compile-flags :cuda :cuda-arch "sm_80") (compile-flags :cuda :cuda-arch "sm_80")))
  (is (member "--iree-cuda-target=sm_80" (compile-flags :cuda :cuda-arch "sm_80") :test #'string=))
  (signals error (compile-flags :nope)))

(define-iree-test compiler/compiler-revision/mentions-locked-commit
    "compiler-revision の文字列は third_party/iree.lock で固定したコミット
ハッシュを含む。"
  (skip-unless-iree :library :compiler)
  (is (search (%locked-iree-commit) (compiler-revision))))

(fiveam:test (library/ensure-compiler-loaded/missing-home-signals-library-not-found :suite :nabla.medium)
  "存在しないディレクトリを NABLA_IREE_HOME として渡したときのパス解決先には
共有ライブラリが無い。ensure-compiler-loaded そのもの（実際にロードして
コンディションを出すところ）は、このプロセスの中で他のテストが一度でも
コンパイラのロードに成功すると *compiler-loaded-p* が真のままになり
（グローバル初期化はプロセスにつき1回だけ、という仕様そのもの）、以後
何もせず即座に返ってしまうため直接は検証できない。そのためロードを
伴わない、純粋なパス解決だけを確かめる。"
  (let ((missing-home (merge-pathnames "nabla-iree-definitely-missing-home/" (nabla.iree::iree-home))))
    (is (not (probe-file missing-home)))
    (is (not (probe-file (nabla.iree::%library-path missing-home :compiler))))
    (is (not (probe-file (nabla.iree::%library-path missing-home :runtime))))))
