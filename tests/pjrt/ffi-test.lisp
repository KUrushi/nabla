;;;; PJRT の FFI 定義（src/pjrt/ffi.lisp）が、固定したヘッダ
;;;; third_party/pjrt/pjrt_c_api.h と食い違っていないことの検査（issue #85）。
;;;; プラグインが無くても動く（small）。構造体の配置やオフセットを
;;;; 記憶ではなくヘッダから写す、という約束を機械的に守らせる。

(in-package #:nabla.pjrt.tests)

(defun %header-text ()
  (let ((path (asdf:system-relative-pathname "nabla/pjrt" "third_party/pjrt/pjrt_c_api.h")))
    (with-open-file (in path)
      (let* ((text (make-string (file-length in)))
             (n (read-sequence text in)))
        (subseq text 0 n)))))

(defun %identifier-char-p (char)
  (or (alphanumericp char) (char= char #\_)))

(defun %header-api-function-names ()
  "ヘッダの struct PJRT_Api の _PJRT_API_STRUCT_FIELD( の直後の識別子を、
宣言順に集める。"
  (let* ((text (%header-text))
         (start (search "typedef struct PJRT_Api {" text))
         (end (search "} PJRT_Api;" text :start2 start))
         (marker "_PJRT_API_STRUCT_FIELD(")
         (names '()))
    (loop with pos = start
          for hit = (search marker text :start2 pos :end2 end)
          while hit
          do (let* ((name-start (position-if-not (lambda (c) (member c '(#\Space #\Newline)))
                                                 text :start (+ hit (length marker))))
                    (name-end (position-if-not #'%identifier-char-p text :start name-start)))
               (push (subseq text name-start name-end) names)
               (setf pos name-end)))
    (nreverse names)))

(defun %header-enum-ordinals (type-name prefix)
  "ヘッダの `typedef enum { ... } TYPE-NAME;` の中で、行頭が PREFIX で始まる
列挙子を、明示的な値なしの宣言順に 0, 1, 2, ... と数えて (名前 . 値) の
リストにする。"
  (let* ((text (%header-text))
         (close (search (format nil "} ~A;" type-name) text))
         (open (search "typedef enum {" text :from-end t :end2 close))
         (ordinals '())
         (next 0))
    (with-input-from-string (in (subseq text open close))
      (loop for line = (read-line in nil)
            while line
            do (let ((trimmed (string-left-trim " " line)))
                 (when (and (>= (length trimmed) (length prefix))
                            (string= prefix trimmed :end2 (length prefix)))
                   (let ((end (position-if-not #'%identifier-char-p trimmed)))
                     (push (cons (subseq trimmed 0 end) next) ordinals)
                     (incf next))))))
    (nreverse ordinals)))

(test (api-function-names-match-the-header :suite :nabla.small)
  "*api-function-names* は、ヘッダの struct PJRT_Api の関数ポインタの並びと
一致する（オフセットはこの並びの位置から計算するため、1つでもずれると
別の関数を呼んでしまう）。"
  (let ((header (%header-api-function-names)))
    (is (plusp (length header)))
    (is (equal header nabla.pjrt::*api-function-names*))))

(test (buffer-types-match-the-header :suite :nabla.small)
  "*buffer-types* の値は、ヘッダの PJRT_Buffer_Type の列挙の位置と一致する。"
  (let ((ordinals (%header-enum-ordinals "PJRT_Buffer_Type" "PJRT_Buffer_Type_")))
    (is (= 0 (cdr (assoc "PJRT_Buffer_Type_INVALID" ordinals :test #'string=))))
    (loop for (dtype . value) in nabla.pjrt::*buffer-types*
          for name = (format nil "PJRT_Buffer_Type_~A"
                             ;; 整数は nabla の dtype タグとヘッダの綴りが違う（issue #126）
                             (case dtype (:i1 "PRED") (:i32 "S32") (:u32 "U32") (:u64 "U64")
                               (t (string-upcase (symbol-name dtype)))))
          do (is (eql value (cdr (assoc name ordinals :test #'string=)))
                 "~A: expected ~A" name (cdr (assoc name ordinals :test #'string=))))))

(test (host-buffer-semantics-match-the-header :suite :nabla.small)
  "+host-buffer-immutable-until-transfer-completes+ は、ヘッダの
PJRT_HostBufferSemantics の列挙の位置と一致する。"
  (let ((ordinals (%header-enum-ordinals "PJRT_HostBufferSemantics" "PJRT_HostBufferSemantics_k")))
    (is (eql nabla.pjrt::+host-buffer-immutable-until-transfer-completes+
             (cdr (assoc "PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes" ordinals
                         :test #'string=))))))

(test (api-functions-offset-matches-the-header-prefix :suite :nabla.small)
  "最初の関数ポインタの位置 +api-functions-offset+ は、%api-head（struct_size,
extension_start, PJRT_Api_Version）の直後。"
  (is (= nabla.pjrt::+api-functions-offset+
         (cffi:foreign-type-size '(:struct nabla.pjrt::%api-head)))))

(test (pjrt-error-is-a-backend-error :suite :nabla.small)
  "pjrt-error は nabla:backend-error の subtype。"
  (is (subtypep 'pjrt-error 'nabla:backend-error))
  (is (subtypep 'pjrt-object-released 'pjrt-error)))
