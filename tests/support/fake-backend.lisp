;;;; フェイク backend: nabla:backend プロトコルの配管（多値、aval、引数の
;;;; 順序、find-backend の同一性、ディスクキャッシュ）を、IREE を経由せずに
;;;; 確かめるための参照実装（issue #9）。
;;;;
;;;; StableHLO の一般的な解釈器ではない。add / dot_general（matmul）/
;;;; reduce（総和）の3つの演算だけを、TEXT の中の固定文字列を SEARCH する
;;;; ことで見分ける。要素型は f32 だけをサポートする（bf16 は IREE 経由で
;;;; しか確かめない。CLAUDE.md / properties.md の「フェイクは振る舞いを
;;;; 持つが本物ではない」の考え方どおり、フェイクどうしで確かめる性質は
;;;; プロトコルの配管と #10 のディスクキャッシュだけに使う）。

(in-package #:nabla.tests.support)

(define-condition %fake-backend-unsupported-op (nabla:backend-error simple-error)
  ()
  (:documentation
   "FAKE-BACKEND の BACKEND-COMPILE / BACKEND-LOAD に、add / dot_general /
reduce のどれも含まない TEXT を渡したときに signal する（IREE の実際の
コンパイルエラーの代わり）。"))

(defclass fake-backend (nabla:backend)
  ((target :initarg :target :initform :fake :reader nabla:backend-target)
   (fingerprint :initarg :fingerprint :initform (list "fake")
                :reader nabla:backend-fingerprint)
   (compile-count :initform 0 :accessor fake-backend-compile-count
                  :documentation "BACKEND-COMPILE を呼んだ回数。0から始まる。"))
  (:documentation
   "NABLA:BACKEND のフェイク実装。IREE を経由せず、Lisp の中だけで
add / dot_general（matmul）/ reduce（総和）の3つの演算を素朴に計算する。
MAKE-BACKEND :FAKE で作る。"))

(defmethod nabla:make-backend ((kind (eql :fake)) &key (fingerprint (list "fake")) (target :fake))
  (make-instance 'fake-backend :target target :fingerprint fingerprint))

(defstruct (fake-module (:constructor %make-fake-module (kind &key reduce-axis)))
  "FAKE-BACKEND の BACKEND-LOAD が返す不透明な module。KIND は :add /
:matmul / :reduce-sum のどれか。REDUCE-AXIS は :reduce-sum のときだけ使う。"
  kind
  reduce-axis)

(defclass fake-array ()
  ((data :initarg :data :accessor %fake-array-data)
   (aval :initarg :aval :reader nabla:device-array-aval))
  (:documentation
   "FAKE-BACKEND の TO-DEVICE / BACKEND-INVOKE が返す device array 相当の
クラス。DATA は Lisp の多次元配列（コピー済み）をそのまま持つ。"))

(defun %search-reduce-axis (text)
  "TEXT の中の \"dimensions = array<i64: N>\" から N（整数）を取り出す。
見つからなければエラーを signal する。"
  (let ((marker "dimensions = array<i64:"))
    (let ((position (search marker text)))
      (unless position
        (error "fake-backend: reduce の dimensions が見つからない: ~S" text))
      (or (parse-integer text :start (+ position (length marker)) :junk-allowed t)
          (error "fake-backend: reduce の dimensions を解析できない: ~S" text)))))

(defun %fake-backend-classify (text)
  "TEXT に含まれる演算を :add / :matmul / :reduce-sum のどれかに分類する。
reduce の本体には add も含まれるため、dot_general → reduce → add の順に
調べる（この順序が重要）。どれにも一致しなければ NIL。"
  (cond
    ((search "stablehlo.dot_general" text) :matmul)
    ((search "stablehlo.reduce" text) :reduce-sum)
    ((search "stablehlo.add" text) :add)
    (t nil)))

(defmethod nabla:backend-compile ((backend fake-backend) text)
  "COMPILE-COUNT を1増やしてから、TEXT を分類できるか確かめる（実際の
コンパイルはしない）。TEXT の UTF-8 バイト列をそのまま「コンパイル済み
モジュール」として返す。add / dot_general / reduce のどれも含まなければ
%FAKE-BACKEND-UNSUPPORTED-OP を signal する。"
  (incf (fake-backend-compile-count backend))
  (unless (%fake-backend-classify text)
    (error '%fake-backend-unsupported-op
           :format-control "fake-backend: TEXT に add / dot_general / reduce のどれも見つからない: ~S"
           :format-arguments (list text)))
  (sb-ext:string-to-octets text :external-format :utf-8))

(defmethod nabla:backend-load ((backend fake-backend) octets)
  "OCTETS を UTF-8 の文字列に戻し、%FAKE-BACKEND-CLASSIFY で分類した
FAKE-MODULE を返す。"
  (declare (ignore backend))
  (let* ((text (sb-ext:octets-to-string octets :external-format :utf-8))
         (kind (%fake-backend-classify text)))
    (unless kind
      (error '%fake-backend-unsupported-op
             :format-control "fake-backend: TEXT に add / dot_general / reduce のどれも見つからない: ~S"
             :format-arguments (list text)))
    (if (eq kind :reduce-sum)
        (%make-fake-module :reduce-sum :reduce-axis (%search-reduce-axis text))
        (%make-fake-module kind))))

(defmethod nabla:backend-unload ((backend fake-backend) module)
  (declare (ignore backend module))
  (values))

(defun %coerce-to-single-float-array (array)
  "ARRAY（DOUBLE-FLOAT の配列、reference-* の返り値）と同じ shape の
SINGLE-FLOAT 配列にコピーして返す（フェイクは f32 だけサポートする）。"
  (let ((result (make-array (array-dimensions array) :element-type 'single-float)))
    (dotimes (i (array-total-size array) result)
      (setf (row-major-aref result i) (coerce (row-major-aref array i) 'single-float)))))

(defmethod nabla:backend-invoke ((backend fake-backend) module function-name &rest arrays)
  "MODULE の KIND に応じて reference-add / reference-matmul /
reference-reduce-sum を ARRAYS（FAKE-ARRAY のリスト）の中身に適用し、
SINGLE-FLOAT にキャストした結果を1つの FAKE-ARRAY にして多値で返す。"
  (declare (ignore backend function-name))
  (let ((inputs (mapcar #'%fake-array-data arrays)))
    (let ((result
            (ecase (fake-module-kind module)
              (:add (reference-add (first inputs) (second inputs)))
              (:matmul (reference-matmul (first inputs) (second inputs)))
              (:reduce-sum (reference-reduce-sum (first inputs) (fake-module-reduce-axis module))))))
      (let ((f32-result (%coerce-to-single-float-array result)))
        (values (make-instance 'fake-array
                                :data f32-result
                                :aval (nabla:array-aval f32-result :f32)))))))

(defmethod nabla:to-device (array (backend fake-backend) &key dtype)
  "ARRAY をコピーして FAKE-ARRAY に包む（IREE 版の TO-DEVICE のフェイク）。

本物の IREE backend と同じく、どの dtype（:f64 と :i1 を含む。issue #72）
の配列もそのまま受け取る。"
  (declare (ignore backend))
  (let ((aval (nabla:array-aval array dtype)))
    (let ((copy (make-array (array-dimensions array) :element-type (array-element-type array))))
      (dotimes (i (array-total-size array))
        (setf (row-major-aref copy i) (row-major-aref array i)))
      (make-instance 'fake-array :data copy :aval aval))))

(defmethod nabla:to-host ((device-array fake-array))
  "FAKE-ARRAY が持つ配列をコピーして返す（IREE 版の TO-HOST のフェイク）。"
  (let* ((data (%fake-array-data device-array))
         (copy (make-array (array-dimensions data) :element-type (array-element-type data))))
    (dotimes (i (array-total-size data) copy)
      (setf (row-major-aref copy i) (row-major-aref data i)))))
