;;;; 並列ローダ（issue #286）の速さを測る。結果は docs/data-loader-parallel.md。
;;;;
;;;;   export CL_SOURCE_REGISTRY="$PWD/:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:"
;;;;   sbcl --non-interactive --load scripts/bench-data-loader.lisp
;;;;
;;;; 32x32 の f32 の画像 NABLA_LOADER_RECORDS 個（既定 1024）の array-source に、1 レコードあたり数 ms の
;;;; 重い map（3x3 の平滑化を何度もかける純 Lisp の計算。割り当ては出力の配列だけ）をかけ、batch-size 32 で
;;;; 1 エポックを読み切る時間を測る。ワーカー数（NABLA_LOADER_WORKERS、既定 "0 1 2 4 8"。0 は逐次）と
;;;; 先読み（prefetch 0 と 2）を変え、さらに「呼び出し側がバッチごとに学習ステップの代わりの計算を
;;;; NABLA_LOADER_STEP_MS（既定 20）ms する」場合も測る（先読みがステップと読み込みを重ねる効果）。
;;;; 時間は NABLA_LOADER_REPS（既定 3）回の中央値（ミリ秒）。1 行 1 レコードの S 式で出す。

(require :asdf)
(asdf:load-system "nabla/data")

(defpackage #:nabla-bench-data-loader (:use #:cl))
(in-package #:nabla-bench-data-loader)

(defun env-integers (name default)
  (let ((value (sb-ext:posix-getenv name)))
    (if (and value (plusp (length value)))
        (with-input-from-string (s value) (loop for x = (read s nil) while x collect x))
        default)))

(defparameter *records* (first (env-integers "NABLA_LOADER_RECORDS" '(1024))))
(defparameter *workers* (env-integers "NABLA_LOADER_WORKERS" '(0 1 2 4 8)))
(defparameter *step-ms* (first (env-integers "NABLA_LOADER_STEP_MS" '(20))))
(defparameter *reps* (first (env-integers "NABLA_LOADER_REPS" '(3))))
(defparameter *smoothing-passes* 40
  "1 レコードあたりの平滑化の回数（手元の Mac で 1 レコード数 ms になる大きさ）。")

(defun smooth (image passes)
  "IMAGE（32x32 の single-float）に 3x3 の平均を PASSES 回かけた新しい配列。"
  (declare (type (simple-array single-float (32 32)) image) (type fixnum passes)
           (optimize (speed 3) (safety 0)))
  (let ((a (make-array '(32 32) :element-type 'single-float))
        (b (make-array '(32 32) :element-type 'single-float)))
    (replace (sb-ext:array-storage-vector a) (sb-ext:array-storage-vector image))
    (dotimes (pass passes a)
      (dotimes (i 32)
        (dotimes (j 32)
          (let ((sum 0f0) (count 0f0))
            (declare (type single-float sum count))
            (loop for di fixnum from -1 to 1
                  do (loop for dj fixnum from -1 to 1
                           do (let ((y (+ i di)) (x (+ j dj)))
                                (when (and (< -1 y 32) (< -1 x 32))
                                  (incf sum (aref a y x))
                                  (incf count 1f0)))))
            (setf (aref b i j) (/ sum count)))))
      (rotatef a b))))

(defun heavy-map ()
  (nabla.data:map-transform
   (lambda (record)
     (list (smooth (first record) *smoothing-passes*) (second record)))))

(defun make-source ()
  (let ((images (make-array (list *records* 32 32) :element-type 'single-float))
        (labels (make-array *records* :element-type '(signed-byte 32))))
    (dotimes (i (array-total-size images))
      (setf (row-major-aref images i) (float (mod (* i 7919) 251) 1f0)))
    (dotimes (i *records*) (setf (aref labels i) (mod i 10)))
    (nabla.data:make-array-source (list images labels))))

(defun busy-ms (ms)
  "呼び出し側のスレッドで MS ミリ秒ぶん計算する（学習ステップの代わり）。"
  (let ((end (+ (get-internal-real-time) (* ms (/ internal-time-units-per-second 1000)))))
    (loop while (< (get-internal-real-time) end))))

(defun epoch-ms (loader workers prefetch step-ms)
  (let ((start (get-internal-real-time)))
    (nabla.data:with-data-iterator (it loader :num-workers workers :prefetch prefetch)
      (loop for (batch more) = (multiple-value-list (nabla.data:next-batch it))
            while more do (busy-ms step-ms)))
    (/ (- (get-internal-real-time) start) (/ internal-time-units-per-second 1000d0))))

(defun median (samples)
  (nth (floor (length samples) 2) (sort (copy-list samples) #'<)))

(let* ((loader (nabla.data:data-loader (make-source) :batch-size 32 :num-epochs 1
                                                     :transforms (list (heavy-map))))
       (per-record (let ((record (nabla.data:source-item (make-source) 0))
                         (start (get-internal-real-time)))
                     (dotimes (i 20) (smooth (first record) *smoothing-passes*))
                     (/ (- (get-internal-real-time) start) (/ internal-time-units-per-second 1000d0) 20))))
  (format t "(:records ~D :batch-size 32 :map-ms-per-record ~,2F :step-ms ~D :reps ~D :cores ~D)~%"
          *records* per-record *step-ms* *reps*
          (or (ignore-errors (parse-integer (uiop:run-program '("sysctl" "-n" "hw.ncpu") :output :string)
                                            :junk-allowed t))
              (ignore-errors (parse-integer (uiop:run-program '("nproc") :output :string) :junk-allowed t))))
  (epoch-ms loader 2 2 0)               ; 暖機
  (dolist (step-ms (list 0 *step-ms*))
    (dolist (workers *workers*)
      (dolist (prefetch (if (zerop workers) '(0) '(0 2)))
        (format t "(:step-ms ~D :workers ~D :prefetch ~D :epoch-ms ~,1F)~%"
                step-ms workers prefetch
                (median (loop repeat *reps* collect (epoch-ms loader workers prefetch step-ms))))
        (finish-output)))))
