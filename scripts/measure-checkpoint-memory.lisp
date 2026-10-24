;;;; save-checkpoint / load-checkpoint のピークメモリを測る（issue #244）。
;;;;
;;;; 1回の起動で1つの操作（save か load）だけを測る。測り方と結果の読み方は
;;;; docs/checkpoint-memory.md、全ケースを回すのは scripts/measure-checkpoint-memory.sh。
;;;;
;;;;   sbcl --dynamic-space-size 8GB --non-interactive --eval '(require :asdf)' \
;;;;     --load scripts/measure-checkpoint-memory.lisp
;;;;
;;;; 環境変数:
;;;;   NABLA_CKPT_OP     save / load
;;;;   NABLA_CKPT_MB     f32 の葉の合計の大きさ（MiB）
;;;;   NABLA_CKPT_SHAPE  few（同じ大きさの葉 4 枚）/ mixed（大きな葉 4 枚で 3/4、残りを 4096 枚の小さな葉）
;;;;   NABLA_CKPT_FILE   save の書き先 / load の読み元（load の前に save で作っておく）
;;;;
;;;; 結果は 1 行の S 式（plist）を標準出力に出す。
;;;;   :rss-base-mb   木を作って full GC した直後の現在の RSS（ps の値）
;;;;   :rss-peak-mb   操作のあとの最大 RSS（getrusage の ru_maxrss。macOS は byte）。ピークの指標はこちら
;;;;   :rss-extra-mb  peak - base（操作が上乗せしたメモリ）。ru_maxrss はプロセスの一生の最大なので、
;;;;                  ロード時の山（約 300 MiB）より小さい操作は peak がその山のままになり、extra は
;;;;                  実際より小さく出る。読めるのは 256 MiB 以上のケース
;;;;   :heap-peak-mb  操作中の GC のたびに記録した dynamic-usage の最大（GC の間の山は拾えない）
;;;;   :status        :ok / :heap-exhausted（dynamic-space-size を超えた）
;;;; save の木は初期化せずに作る（値は 0）。遅延ゼロページなので木のページは作った時点では RSS に載らず、
;;;; 保存中にタッチされて載る。だから save の :rss-base-mb は小さく出る（倍率はピーク RSS で見る）。

(require :sb-posix)
(asdf:load-system "nabla/nn")
(in-package #:nabla.nn)

(defun env (name default)
  (or (sb-ext:posix-getenv name) default))

(defun rss-mb ()
  "このプロセスの最大 RSS（MiB）。macOS の ru_maxrss は byte、Linux は KiB。"
  (let ((raw (nth-value 3 (sb-unix:unix-getrusage sb-unix:rusage_self))))
    (/ raw (if (find :darwin *features*) 1048576.0 1024.0))))

(defun rss-now-mb ()
  "このプロセスの現在の RSS（MiB。ps の値）。"
  (let ((out (with-output-to-string (stream)
               (sb-ext:run-program "/bin/ps" (list "-o" "rss=" "-p" (princ-to-string (sb-posix:getpid)))
                                   :output stream))))
    (/ (parse-integer out :junk-allowed t) 1024.0)))

(defun leaf-sizes (mb shape)
  "合計 MB MiB の f32 の葉ごとの要素数のリスト。"
  (let ((total (floor (* mb 1048576) 4)))
    (ecase (intern (string-upcase shape) :keyword)
      (:few (loop repeat 4 collect (floor total 4)))
      (:mixed (let* ((big (floor (* total 3/4) 4))
                     (small (floor (- total (* 4 big)) 4096)))
                (append (loop repeat 4 collect big)
                        (loop repeat 4096 collect small)))))))

(defvar *heap-peak* 0)

(defun main ()
  (let* ((op (intern (string-upcase (env "NABLA_CKPT_OP" "save")) :keyword))
         (mb (parse-integer (env "NABLA_CKPT_MB" "64")))
         (shape (env "NABLA_CKPT_SHAPE" "few"))
         (file (env "NABLA_CKPT_FILE" "/tmp/ckpt-measure.safetensors"))
         (status :ok) base)
    (push (lambda () (setf *heap-peak* (max *heap-peak* (sb-kernel:dynamic-usage))))
          sb-ext:*after-gc-hooks*)
    ;; load は値を使わないので、ひな形は aval にして木の分のメモリを持たない。
    ;; save の木は初期化せずに確保だけする（値は全部 0）
    (let ((tree (loop for size in (leaf-sizes mb shape)
                      collect (if (eq op :load)
                                  (nabla:make-aval (list size) :f32)
                                  (make-array size :element-type 'single-float :initial-element 0.0)))))
      (sb-ext:gc :full t)
      (setf *heap-peak* (sb-kernel:dynamic-usage)
            base (rss-now-mb))
      (let ((start (get-internal-real-time)) (gc-start sb-ext:*gc-run-time*))
        (handler-case
            (ecase op
              (:save (save-checkpoint file tree))
              (:load (load-checkpoint file tree)))
          (storage-condition () (setf status :heap-exhausted)))
        (let ((peak (rss-mb)))
          (let ((*print-pretty* nil)) (format t "~S~%"
                  (list :op op :mb mb :shape (intern (string-upcase shape) :keyword) :status status
                        :dynamic-space-mb (floor (sb-ext:dynamic-space-size) 1048576)
                        :rss-base-mb (round base) :rss-peak-mb (round peak)
                        :rss-extra-mb (round (- peak base))
                        :heap-peak-mb (round (/ *heap-peak* 1048576.0))
                        :seconds (/ (round (* 100 (/ (- (get-internal-real-time) start)
                                                     internal-time-units-per-second)))
                                    100.0)
                        :gc-seconds (/ (round (* 100 (/ (- sb-ext:*gc-run-time* gc-start)
                                                        internal-time-units-per-second)))
                                       100.0)))))))))

(main)
