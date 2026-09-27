;;;; IREE-BACKEND: core の backend プロトコル（src/backend.lisp、issue #9）
;;;; を実装する、IREE 向けの NABLA:BACKEND。
;;;;
;;;; device（iree_hal_device_t）は生成時には作らない。生成時に作ると、GPU の
;;;; 無いマシンでは :cuda 向けの IREE-BACKEND を作ること自体ができなくなり、
;;;; 「コンパイルだけ」なら GPU なしでも試せるという性質（compile-flags /
;;;; compile-stablehlo は device を必要としない）を失う。そのため device は
;;;; 最初の BACKEND-LOAD / TO-DEVICE のときに、%IREE-BACKEND-DEVICE-LOCK で
;;;; 保護しながら1回だけ作る。GPU の無いマシンで :cuda の IREE-BACKEND に
;;;; BACKEND-LOAD / TO-DEVICE すると、そのときになって MAKE-DEVICE の
;;;; IREE-STATUS-ERROR がそのまま出る。

(in-package #:nabla.iree)

(defun %machine-version ()
  "/proc/cpuinfo の最初の \"model name\" 行の値を返す（読めなければ
\"unknown\"）。:local ターゲットの vmfb は
--iree-llvmcpu-target-cpu=host でコンパイルするため、実行するマシンの
CPU に依存する。BACKEND-FINGERPRINT に含めて、キャッシュキーへ反映させる
のに使う。"
  (handler-case
      (with-open-file (stream #P"/proc/cpuinfo" :direction :input)
        (loop for line = (read-line stream nil nil)
              while line
              when (and (>= (length line) 10) (string= "model name" line :end2 10))
                do (let ((colon (position #\: line)))
                     (return (if colon
                                 (string-trim '(#\Space #\Tab) (subseq line (1+ colon)))
                                 "unknown")))
              finally (return "unknown")))
    (error () "unknown")))

(defclass iree-backend (nabla:backend)
  ((target :initarg :target :reader nabla:backend-target
           :documentation ":local または :cuda。")
   (cuda-arch :initarg :cuda-arch :initform nil :reader %iree-backend-cuda-arch
              :documentation "\"sm_80\" のような文字列、または NIL（IREE の既定）。")
   (device :initform nil :accessor %iree-backend-device
           :documentation "遅延生成する nabla.iree:device。最初の BACKEND-LOAD /
TO-DEVICE まで NIL のまま。")
   (device-lock :initform (sb-thread:make-mutex :name "nabla-iree-backend-device")
                :reader %iree-backend-device-lock
                :documentation "DEVICE の遅延生成を1回だけにするロック。"))
  (:documentation
   "NABLA:BACKEND の IREE 実装。TARGET が :local ならホスト CPU
（llvm-cpu、target-cpu=host）、:cuda なら CUDA-ARCH（例 \"sm_80\"、NIL なら
IREE の既定）向けにコンパイル・実行する。MAKE-BACKEND :IREE で作る。"))

(defmethod nabla:make-backend ((kind (eql :iree)) &key (target :local) cuda-arch)
  "TARGET（:local または :cuda）向けの IREE-BACKEND を作る。生成時に
compiler / runtime それぞれの共有ライブラリの有無を probe し、無い方に
ついて（両方無ければ compiler の方を先に）IREE-LIBRARY-NOT-FOUND を
signal する。device はまだ作らない（ファイル先頭のコメント参照）。"
  (ecase target
    (:local)
    (:cuda))
  (let ((home (iree-home)))
    (unless (iree-available-p :library :compiler)
      (error 'iree-library-not-found
             :path (%library-path home :compiler) :home home :library :compiler))
    (unless (iree-available-p :library :runtime)
      (error 'iree-library-not-found
             :path (%library-path home :runtime) :home home :library :runtime)))
  (make-instance 'iree-backend :target target :cuda-arch cuda-arch))

(defun %iree-backend-ensure-device (backend)
  "BACKEND の device を、まだ無ければ (MAKE-DEVICE (BACKEND-TARGET BACKEND))
で作って返す（1回だけ、%IREE-BACKEND-DEVICE-LOCK で保護する）。

浮動小数点トラップのマスクは MAKE-DEVICE 自身の責任（runtime.lisp）。ここでは
何もしない（MAKE-DEVICE を直接呼ぶ他の呼び出し元も同じ保護を受けるように、
IREE-BACKEND 経由の呼び出しだけをここで包まない）。"
  (sb-thread:with-mutex ((%iree-backend-device-lock backend))
    (or (%iree-backend-device backend)
        (setf (%iree-backend-device backend)
              (make-device (nabla:backend-target backend))))))

(defmethod nabla:backend-fingerprint ((backend iree-backend))
  "BACKEND-COMPILE の出力を左右するものすべて（コンパイラのリビジョン、
target、cuda-arch、解決済みのコンパイルフラグ、:local なら CPU の
model name）を文字列のリストにして返す。COMPILE-FLAGS は毎回呼び直す
（iree-lld の有無のような、実行するマシンによって動的に変わりうる部分を
反映するため）。"
  (let* ((target (nabla:backend-target backend))
         (cuda-arch (%iree-backend-cuda-arch backend)))
    (list* "nabla-module-cache-v1" "iree" (compiler-revision)
           (format nil "target=~(~A~)" target)
           (format nil "cuda-arch=~A" (or cuda-arch "-"))
           (append (compile-flags target :cuda-arch cuda-arch)
                   (when (eq target :local)
                     (list (format nil "host=~A" (%machine-version))))))))

(defmethod nabla:backend-compile ((backend iree-backend) text)
  "TEXT を BACKEND の target / cuda-arch から求めた COMPILE-FLAGS で
COMPILE-STABLEHLO する。"
  (compile-stablehlo text :flags (compile-flags (nabla:backend-target backend)
                                                 :cuda-arch (%iree-backend-cuda-arch backend))))

(defstruct (iree-module (:constructor %make-iree-module (session)))
  "BACKEND-LOAD が返す、IREE-BACKEND にとっての不透明な module。SESSION は
その module を読み込んだ nabla.iree:session。"
  session)

(defmethod nabla:backend-load ((backend iree-backend) octets)
  "BACKEND の device（無ければ遅延生成する）に新しい session を作り、
OCTETS（BACKEND-COMPILE が返した vmfb のバイト列）を
SESSION-APPEND-MODULE で読み込んで IREE-MODULE に包んで返す。途中で
失敗したら、作りかけの session を解放してから再度 signal する。"
  (let ((session (make-session (%iree-backend-ensure-device backend))))
    (handler-case
        (progn
          (session-append-module session octets)
          (%make-iree-module session))
      (error (condition)
        (release-session session)
        (error condition)))))

(defmethod nabla:backend-unload ((backend iree-backend) module)
  "MODULE（BACKEND-LOAD が返した IREE-MODULE）の session を解放する。
すでに解放済みなら何もしない（idempotent）。"
  (declare (ignore backend))
  (unless (session-released-p (iree-module-session module))
    (release-session (iree-module-session module))))

(defmethod nabla:backend-invoke ((backend iree-backend) module function-name &rest arrays)
  "MODULE の session に対して、FUNCTION-NAME の前に固定のモジュール名
\"module\" を付けた \"module.FUNCTION-NAME\" を INVOKE する（フェーズ1の
StableHLO 出力も無名の builtin.module にする前提。無名モジュールは
IREE 側で \"module\" という名前になる）。ARRAYS はそのまま INVOKE に渡す。

浮動小数点トラップのマスクは INVOKE 自身の責任（execute.lisp）。"
  (declare (ignore backend))
  (apply #'invoke (iree-module-session module) (format nil "module.~A" function-name) arrays))

(defmethod to-device (array (backend iree-backend) &key dtype)
  "ARRAY を BACKEND の device（無ければ遅延生成する）にコピーする。実際の
コピーは device に特化した TO-DEVICE メソッド（device-array.lisp）へ
委譲する。"
  (to-device array (%iree-backend-ensure-device backend) :dtype dtype))
