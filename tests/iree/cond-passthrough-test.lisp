;;;; stablehlo.if の枝が入力をそのまま返す出力（素通し）の IREE 3.11 のバグと、その回避策
;;;; （issue #226、docs/iree-upstream-bugs.md の 8、medium）。
;;;;
;;;; 守らせる性質:
;;;; - バグが IREE に残っていること（直ったら失敗する。%cond-emit の回避策を外す合図）: 最小の再現
;;;;   docs/iree-repros/if-passthrough-operands.mlir を iree-run-module に流すと、素通しでない側の枝
;;;;   を選んだとき "ref is null" で失敗する（local-task も local-sync も。もう片方の枝は通る）。
;;;;   対照の if-passthrough-operands-select.mlir は両方の枝で通る。呼び出しは子プロセスで行う。
;;;; - 回避策: 素通しの出力を持つ cond* を IREE で jit した結果は、pred が真でも偽でも eager と一致する
;;;;   （Issue の表の3つの形と、素通しの位置・個数・両枝・出力の個数をランダムにした PBT）。

(in-package #:nabla.iree.tests)

;;; ---- 今も落ちること ----

(defun %repro-path (file)
  (namestring (asdf:system-relative-pathname "nabla" (format nil "docs/iree-repros/~A" file))))

(defun %repro-inputs (file p)
  "FILE の `// INPUT: --input=...` の行のリスト。最初の入力（pred 用の f32）は P に差し替える。"
  (with-open-file (stream (%repro-path file))
    (let ((lines (loop for line = (read-line stream nil)
                       while line
                       when (and (> (length line) 10) (string= "// INPUT: " line :end2 10))
                         collect (subseq line 10))))
      (cons (format nil "--input=f32=~A" p) (rest lines)))))

(defun %run-repro (file device p)
  "docs/iree-repros/FILE を iree-compile し、iree-run-module --device=DEVICE で子プロセスとして
実行して、(values 終了コード 出力) を返す。コンパイルに失敗したら :compile-failed。"
  (let* ((vmfb (namestring (uiop:tmpize-pathname #p"/tmp/nabla-if-passthrough.vmfb")))
         (compile-code
           (nth-value 2 (uiop:run-program
                         (list (%iree-bin "iree-compile")
                               "--iree-hal-target-device=local"
                               "--iree-hal-local-target-device-backends=llvm-cpu"
                               (%repro-path file) "-o" vmfb)
                         :ignore-error-status t :output nil :error-output nil))))
    (unwind-protect
         (if (zerop compile-code)
             (multiple-value-bind (out err code)
                 (uiop:run-program
                  (append (list (%iree-bin "iree-run-module") (format nil "--device=~A" device)
                                (format nil "--module=~A" vmfb) "--function=main")
                          (%repro-inputs file p))
                  :ignore-error-status t :output :string :error-output :string)
               (values code (concatenate 'string out err)))
             :compile-failed)
      (ignore-errors (delete-file vmfb)))))

(define-iree-test cond-passthrough/iree-known-bug-passthrough-operands-fail-at-run-time
    "IREE 3.11 のバグが残っていること: then が3つの入力を素通しする stablehlo.if は、else を選ぶ（p = 1.0）と
\"ref is null\" で失敗する。then を選ぶ（p = 0.0）と通る。local-task も local-sync も。素通しを if の外の
select にした対照は、どちらの枝も通る。直ると失敗する——そのときは %cond-emit（src/primitives/cond.lisp）の
回避策と docs/iree-upstream-bugs.md の 8 を外す。"
  (skip-unless-iree :library :both)
  (unless (and (probe-file (%iree-bin "iree-run-module"))
               (probe-file (%iree-bin "iree-compile")))
    (skip "iree-compile / iree-run-module が無い")
    (return-from iree-test))
  (dolist (device '("local-task" "local-sync"))
    (is (eql 0 (%run-repro "if-passthrough-operands.mlir" device "0.0")) "~A: p=0.0 は通るはず" device)
    (multiple-value-bind (code output) (%run-repro "if-passthrough-operands.mlir" device "1.0")
      (is (and (integerp code) (/= 0 code) (search "ref is null" output))
          "~A: p=1.0 で \"ref is null\" にならなかった（バグが直った？）: ~S ~A" device code output))
    (dolist (p '("0.0" "1.0"))
      (is (eql 0 (%run-repro "if-passthrough-operands-select.mlir" device p))
          "~A: 対照が p=~A で失敗した" device p))))

;;; ---- 回避策 ----

(defparameter *cp-scale* (nb:with-tracing (x) (* 0.9 x)))
(defparameter *cp-lt-half* (nb:with-tracing (p) (< p 0.5)))

(defun %cp-operands (shape)
  "5個の operands（シード固定）。"
  (loop for i below 5
        collect (make-random-array (make-array-spec shape :f32) :seed (+ 100 i))))

(defun %cp-branch (specs)
  "SPECS（出力ごとに (:pass i) か (:compute i)）の枝。(:pass i) は i 番目の operand をそのまま返し、
(:compute i) は 0.9 倍して返す。"
  (nb::%make-traceable-function
   (loop for i below 5 collect (intern (format nil "X~D" i)))
   (lambda (&rest xs)
     (values-list (mapcar (lambda (spec)
                            (destructuring-bind (kind i) spec
                              (if (eq kind :pass) (nth i xs) (funcall *cp-scale* (nth i xs)))))
                          specs)))))

(defun %cp-jit-matches-eager-p (then-specs else-specs backend &key (shape '()))
  "THEN-SPECS / ELSE-SPECS の枝の cond* を pred の両方の値で jit し、eager と比べる。"
  (let ((f (nb::%make-traceable-function
            (list 'p 'x0 'x1 'x2 'x3 'x4)
            (lambda (p &rest xs)
              (apply #'nb:cond* (funcall *cp-lt-half* p) (%cp-branch then-specs) (%cp-branch else-specs) xs)))))
    (every (lambda (p)
             (%pc-jit-matches-eager-p f backend (cons (nb::%scalar-array p :f32) (%cp-operands shape))))
           '(0.0 1.0))))

(define-iree-test cond-passthrough/jit-issue-table-cases-match-eager
    "Issue #226 の表の形（素通し1つ + 計算した3つで失敗、素通し1つ + 計算した2つ、両枝が同じ operand を素通し）と、
then が3つ素通しする形を、スカラーと [2 3] の配列で jit した結果が eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (dolist (shape '(() (2 3)))
      ;; 失敗した形: then (0.9a, b, c, d) / else (a, 0.9b, 0.9c, 0.9d)
      (is (%cp-jit-matches-eager-p '((:compute 0) (:pass 1) (:pass 2) (:pass 3))
                                   '((:pass 0) (:compute 1) (:compute 2) (:compute 3))
                                   backend :shape shape)
          "素通し1つ + 計算した3つ（形 ~A）" shape)
      ;; 通った形: オペランド3つ
      (is (%cp-jit-matches-eager-p '((:compute 0) (:pass 1) (:pass 2))
                                   '((:pass 0) (:compute 1) (:compute 2))
                                   backend :shape shape)
          "素通し1つ + 計算した2つ（形 ~A）" shape)
      ;; 両枝が同じ operand を素通し
      (is (%cp-jit-matches-eager-p '((:compute 0) (:pass 1))
                                   '((:compute 1) (:pass 1))
                                   backend :shape shape)
          "両枝が同じオペランドを素通し（形 ~A）" shape)
      ;; 最小の再現（then が3つ素通し、else は全て計算）
      (is (%cp-jit-matches-eager-p '((:compute 0) (:pass 1) (:pass 2) (:pass 3))
                                   '((:compute 0) (:compute 1) (:compute 2) (:compute 3))
                                   backend :shape shape)
          "then が3つ素通し（形 ~A）" shape))
    (gc-and-run-finalizers)))

(define-iree-test cond-passthrough/jit-branch-returning-captured-arrays-matches-eager
    "枝が閉包で捕まえた外側の配列（cond* の eqn の追加の入力になる）をそのまま返す形も、jit した結果が
pred の両方の値で eager と一致する（then が捕捉値を3つ素通し、else は全て計算）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (dolist (shape '(() (2 3)))
      (let ((f (nb::%make-traceable-function
                (list 'p 'x0 'x1 'x2 'x3)
                (lambda (p x0 x1 x2 x3)
                  (nb:cond* (funcall *cp-lt-half* p)
                            (nb::%make-traceable-function
                             (list 'a) (lambda (a) (values (funcall *cp-scale* a) x1 x2 x3)))
                            (nb::%make-traceable-function
                             (list 'a) (lambda (a) (values (funcall *cp-scale* a) (funcall *cp-scale* x1)
                                                           (funcall *cp-scale* x2) (funcall *cp-scale* x3))))
                            x0)))))
        (dolist (p '(0.0 1.0))
          (is (%pc-jit-matches-eager-p f backend (cons (nb::%scalar-array p :f32) (subseq (%cp-operands shape) 0 4)))
              "捕捉値の素通し（形 ~A、p ~A）" shape p))))
    (gc-and-run-finalizers)))

(defparameter *cp-add-self* (nb:with-tracing (x) (+ x x)))

(define-iree-test cond-passthrough/jit-i32-passthrough-operands-match-eager
    "i32 の operand を素通しする cond*（then が3つ素通し、else は全て計算）も、jit した結果が pred の両方の値で
eager と完全に一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil)
        (f (nb::%make-traceable-function
            (list 'p 'x0 'x1 'x2 'x3)
            (lambda (p x0 x1 x2 x3)
              (nb:cond* (funcall *cp-lt-half* p)
                        (nb::%make-traceable-function
                         (list 'a 'b 'c 'd) (lambda (a b c d) (values (funcall *cp-add-self* a) b c d)))
                        (nb::%make-traceable-function
                         (list 'a 'b 'c 'd) (lambda (a b c d) (values (funcall *cp-add-self* a) (funcall *cp-add-self* b)
                                                                      (funcall *cp-add-self* c) (funcall *cp-add-self* d))))
                        x0 x1 x2 x3)))))
    (let ((jf (nb:jit f :backend backend))
          (xs (loop for i below 4
                    collect (make-random-array (make-array-spec '(3) :i32) :seed (+ 200 i)))))
      (unwind-protect
           (dolist (p '(0.0 1.0))
             (let ((args (cons (nb::%scalar-array p :f32) xs)))
               (is (every #'equalp (multiple-value-list (apply jf args)) (multiple-value-list (apply f args)))
                   "i32 の素通し（p ~A）" p)))
        (nb::%jit-cache-forget f)))
    (gc-and-run-finalizers)))

(defparameter *cp-spec-generator*
  (generator (tuple (uniform-integer :lo 1 :hi 6)
                    (list (tuple (uniform-integer :lo 0 :hi 2) (uniform-integer :lo 0 :hi 4)) :min-length 6 :max-length 6)
                    (list (tuple (uniform-integer :lo 0 :hi 2) (uniform-integer :lo 0 :hi 4)) :min-length 6 :max-length 6)))
  "(出力の個数 then の6個の spec else の6個の spec)。spec は (0=計算 / 1,2=素通し, operand の番号)。素通しが多めに出る。")

(define-iree-test cond-passthrough/jit-random-passthrough-patterns-match-eager
    "素通しの位置・個数・then / else のどちらか・両方、出力の個数（1〜6）をランダムにした cond* を jit した結果は、
pred の両方の値で eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 30)
        (nb:*compile-cache-directory* nil))
    (flet ((specs (count raw)
             (loop for (kind i) in (subseq raw 0 count)
                   collect (list (if (>= kind 1) :pass :compute) i))))
      (is (check-it *cp-spec-generator*
                    (lambda (case)
                      (destructuring-bind (count then-raw else-raw) case
                        (%cp-jit-matches-eager-p (specs count then-raw) (specs count else-raw) backend
                                                 :shape '(3))))
                    :regression-id cond-passthrough/jit-random-passthrough-patterns-match-eager
                    :regression-file (regression-path "iree-cond-passthrough-random"
                                                      :package "NABLA.IREE.TESTS"))))
    (gc-and-run-finalizers)))
