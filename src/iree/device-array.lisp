;;;; device-array: Lisp の配列と IREE の buffer view を橋渡しする、JAX の
;;;; jax.Array に相当するクラス（issue #7）。
;;;;
;;;; nabla.iree は nabla を :use しないので、nabla:make-aval のように常に
;;;; パッケージ名を書く。ただし TO-DEVICE / TO-HOST / DEVICE-ARRAY-AVAL は
;;;; core の backend プロトコル（issue #9、src/backend.lisp）の総称関数を
;;;; src/iree/package.lisp で import-from しており、この3つだけは
;;;; パッケージ名を付けずに書く（下の DEFMETHOD / :reader がそのまま
;;;; nabla:to-device などへメソッドを追加する）。
;;;;
;;;; メモリの所有権: device-array は生成時（%wrap-buffer-view）に、渡された
;;;; buffer view の所有権を引き取り、同時に DEVICE の iree_hal_device_t を
;;;; retain する。retain するのは、IREE の heap buffer が確保元の allocator
;;;; の統計ブロックへの生ポインタを持ち、その allocator は device が所有し
;;;; ているため（buffer_heap.c / task_device.c の実装）。device 側を先に
;;;; release すると、この device-array を解放するときに use-after-free に
;;;; なる。そのため release-device-array は buffer view → device の順で
;;;; 解放する。この retain のおかげで、呼び出し側が device オブジェクトを
;;;; release-device した後でも、生きている device-array の to-host は
;;;; 正しく動き続ける。
;;;;
;;;; 解放の自動化（finalizer、#11）: %wrap-buffer-view は、上の retain の
;;;; 直後に trivial-garbage:finalize で buffer view と device の解放を
;;;; 登録する。finalizer のクロージャは device-array 本体ではなく、
;;;; cffi:pointer-address で整数に変えた2つのポインタの値だけを捕まえる
;;;; （device-array を捕まえると、finalizer 自身がそのオブジェクトへの
;;;; 参照になり、GC が永遠に回収できなくなる。CLAUDE.md の約束）。
;;;; release-device-array は tg:cancel-finalization を先に呼んでから
;;;; 自分で解放するので、明示的に解放した後に GC が走っても二重解放には
;;;; ならない。

(in-package #:nabla.iree)

(defclass device-array ()
  ((pointer :accessor %device-array-pointer :initarg :pointer
            :documentation "iree_hal_buffer_view_t*。release-device-array の
後は null-pointer になる。")
   (device-pointer :accessor %device-array-device-pointer :initarg :device-pointer
                    :documentation "生成時に retain した iree_hal_device_t*。
release-device-array の後は null-pointer になる。")
   (aval :reader device-array-aval :initarg :aval
         :documentation "この device-array の形状と dtype（NABLA:AVAL）。
DEVICE-ARRAY-AVAL は NABLA:DEVICE-ARRAY-AVAL を import-from したシンボルな
ので、この :reader は core の backend プロトコル（issue #9）の総称関数へ
メソッドを追加する。")
   (device :reader device-array-device :initarg :device
           :documentation "この device-array を作った nabla.iree:device
オブジェクト（#8 の invoke が session の device と eq かどうかを比べるのに
使う）。"))
  (:documentation
   "IREE デバイス上の buffer view を包む、JAX の jax.Array に相当するクラス
（用語集参照）。TO-DEVICE / (#8 の) INVOKE の出力としてのみ作られ、直接
MAKE-INSTANCE することは想定していない。"))

(defun %wrap-buffer-view (buffer-view device aval)
  "BUFFER-VIEW（foreign pointer、iree_hal_buffer_view_t*）の所有権を
引き取って DEVICE-ARRAY に包む唯一のコンストラクタ。DEVICE の
iree_hal_device_t を iree_hal_device_retain で retain し、その pointer も
一緒に保持する（ファイル先頭のコメントの理由）。TO-DEVICE と、#8 の
INVOKE の出力の組み立てがここを通る。

生成した DEVICE-ARRAY には trivial-garbage:finalize で自動解放を登録する
（#11）。finalizer のクロージャは BUFFER-VIEW と DEVICE-POINTER の
cffi:pointer-address（整数）だけを捕まえ、DEVICE-ARRAY 本体・AVAL・
DEVICE オブジェクトのどれも参照しない。release-device-array で明示的に
解放した場合は tg:cancel-finalization でこの finalizer を先に取り消すので、
二重解放にはならない。"
  (let ((device-pointer (%live-device-pointer device "%wrap-buffer-view")))
    (%hal-device-retain device-pointer)
    (let ((array (make-instance 'device-array
                                 :pointer buffer-view
                                 :device-pointer device-pointer
                                 :aval aval
                                 :device device))
          (buffer-view-address (cffi:pointer-address buffer-view))
          (device-pointer-address (cffi:pointer-address device-pointer)))
      (tg:finalize array
                    (lambda ()
                      (%hal-buffer-view-release (cffi:make-pointer buffer-view-address))
                      (%hal-device-release (cffi:make-pointer device-pointer-address))))
      array)))

(defun device-array-released-p (device-array)
  "DEVICE-ARRAY が release-device-array 済みなら真を返す。"
  (cffi:null-pointer-p (%device-array-pointer device-array)))

(defun %live-device-array-pointer (device-array context)
  "DEVICE-ARRAY の foreign pointer（iree_hal_buffer_view_t*）を返す。
release-device-array 済みなら、解放済みの NULL ポインタを C へ渡して
クラッシュさせる前に IREE-OBJECT-RELEASED（kind :device-array）を signal
する。CONTEXT は呼び出し元の nabla.iree 側の関数名（文字列）。"
  (when (device-array-released-p device-array)
    (error 'iree-object-released :kind :device-array :context context))
  (%device-array-pointer device-array))

;; :i1 のホスト表現（BIT 配列。SBCL は実体をビット詰めで持つ）と IREE の
;; 表現（IREE_HAL_ELEMENT_TYPE_BOOL_8、1要素1バイト。buffer_view.h:140）が
;; 違うので、:i1 だけは sb-ext:array-storage-vector を直接渡せず、行優先の
;; 順に1要素1バイトへ展開・圧縮する（issue #72）。

(defun %i1-octets (array)
  "BIT の配列 ARRAY の要素を行優先の順に並べた (unsigned-byte 8) のベクタ
（1要素1バイト、値は 0 か 1）を新しく作って返す。"
  (let ((octets (make-array (array-total-size array) :element-type '(unsigned-byte 8))))
    (dotimes (i (length octets) octets)
      (setf (aref octets i) (row-major-aref array i)))))

(defun %unpack-i1-octets (octets array)
  "OCTETS（IREE から読み出した1要素1バイトの :i1）の各バイトの最下位ビットを、
BIT の配列 ARRAY に行優先の順で書き込む。"
  (dotimes (i (length octets) array)
    (setf (row-major-aref array i) (logand (aref octets i) 1))))

(defmethod to-device (array (device device) &key dtype)
  "ARRAY（simple-array、rank は任意）を DEVICE 上にコピーし、DEVICE-ARRAY を
返す。コピーは1回だけ行う: sb-ext:array-storage-vector で ARRAY の1次元の
実体ビューを取り、sb-sys:with-pinned-objects でピン留めしたポインタを
buffer-view-allocate-copy に渡す（多次元配列やランク0の配列も
array-storage-vector で1次元ビューにできる。displaced な配列は
array-storage-vector が simple-error を出すので、その前に check-type で
分かりやすい TYPE-ERROR にする）。

:i1 だけは例外で、BIT 配列の実体はビット詰めなのに対して IREE の
IREE_HAL_ELEMENT_TYPE_BOOL_8 は1要素1バイトなので、%I1-OCTETS で
(unsigned-byte 8) のベクタに展開してからそれを渡す（issue #72）。

ARRAY が simple-array でなければ（adjustable / displaced）TYPE-ERROR、
ARRAY の要素型と DTYPE が矛盾すれば NABLA:DTYPE-MISMATCH（NABLA:ARRAY-AVAL
経由）が signal される。デバイスに送れる dtype は *element-types* にある
もの（:f32 / :f64 / :bf16 / :f16 / :i1。issue #72 で :f64 と :i1 を
足した）で、それ以外の dtype には NABLA:UNSUPPORTED-DTYPE を signal する
（issue #37）。DEVICE が release-device 済みなら IREE-OBJECT-RELEASED
（kind :device）が signal される。"
  (check-type array simple-array)
  (let* ((aval (nabla:array-aval array dtype))
         (element-dtype (nabla:aval-dtype aval))
         (storage (if (eq element-dtype :i1)
                      (%i1-octets array)
                      (sb-ext:array-storage-vector array))))
    (unless (assoc element-dtype *element-types*)
      (error 'nabla:unsupported-dtype :dtype element-dtype))
    (sb-sys:with-pinned-objects (storage)
      (let ((buffer-view
              (buffer-view-allocate-copy device (nabla:aval-shape aval) element-dtype
                                          (sb-sys:vector-sap storage)
                                          (nabla:aval-byte-length aval))))
        (%wrap-buffer-view buffer-view device aval)))))

(defmethod to-host ((device-array device-array))
  "DEVICE-ARRAY の内容を、DEVICE-ARRAY の aval と同じ shape・dtype を持つ
新しい多次元 simple-array にコピーして返す（コピーは1回。
sb-ext:array-storage-vector で結果配列の1次元ビューを取り、そこへ直接
読み出す。1次元 vector + shape の組ではなく、多次元配列そのものを返すので
(to-device (to-host x)) がそのまま使える）。:i1 は IREE 側が1要素1バイト
なので、いったん (unsigned-byte 8) のベクタに読み出してから、各バイトの
最下位ビットを BIT 配列に詰める（buffer_view.h の BOOLEAN は「整数として
格納し、1ビットだけが有効」。issue #72）。

DEVICE-ARRAY が release-device-array 済みなら IREE-OBJECT-RELEASED
（kind :device-array）が signal される。読み出しに使う device pointer は
DEVICE-ARRAY 自身が生成時に retain したものなので、TO-DEVICE に渡した
device オブジェクトを呼び出し側が release-device した後でも、この
device-array がまだ生きている限り to-host は動き続ける（ファイル先頭の
コメント参照）。"
  (let* ((buffer-view (%live-device-array-pointer device-array "to-host"))
         (aval (device-array-aval device-array))
         (array (make-array (nabla:aval-shape aval)
                             :element-type (nabla:dtype-element-type (nabla:aval-dtype aval))))
         (storage (if (eq (nabla:aval-dtype aval) :i1)
                      (make-array (array-total-size array) :element-type '(unsigned-byte 8))
                      (sb-ext:array-storage-vector array))))
    (sb-sys:with-pinned-objects (storage)
      (%buffer-view-read-into-sap (%device-array-device-pointer device-array)
                                   buffer-view
                                   (sb-sys:vector-sap storage)
                                   (nabla:aval-byte-length aval)))
    (when (eq (nabla:aval-dtype aval) :i1)
      (%unpack-i1-octets storage array))
    array))

(defun release-device-array (device-array)
  "DEVICE-ARRAY を解放する。まず tg:cancel-finalization で %wrap-buffer-view
が登録した finalizer を取り消し（後から GC が走っても、この解放と finalizer
の両方が同じポインタを解放する二重解放にならないようにする）、それから
buffer view を解放してから、生成時に retain した device を解放する
（この順でなければならない理由はファイル先頭のコメント参照）。二重解放は
idempotent（何もしない）。"
  (unless (device-array-released-p device-array)
    (tg:cancel-finalization device-array)
    (%hal-buffer-view-release (%device-array-pointer device-array))
    (%hal-device-release (%device-array-device-pointer device-array))
    (setf (%device-array-pointer device-array) (cffi:null-pointer))
    (setf (%device-array-device-pointer device-array) (cffi:null-pointer))))

(defmethod print-object ((device-array device-array) stream)
  (print-unreadable-object (device-array stream :type t)
    (if (device-array-released-p device-array)
        (format stream "released")
        (let ((aval (device-array-aval device-array)))
          (format stream "~(~A~)[~{~D~^ ~}] ~A"
                  (nabla:aval-dtype aval)
                  (nabla:aval-shape aval)
                  (device-name (device-array-device device-array)))))))
