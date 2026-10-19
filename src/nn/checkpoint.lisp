;;;; PyTree のチェックポイント: 葉をパス名で平坦化して safetensors に保存・復元する
;;;; （issue #206）。
;;;;
;;;; 決めたこと:
;;;;
;;;; - パス名は "." でつないだ文字列。リストとベクタの子は添字（"layers.0"）、defmodule の
;;;;   モジュールの子はスロット名（"w"。:static は葉でないので出ない）。スロット名は
;;;;   Lisp の既定で大文字になる名前（W）だけを小文字にし、|Dense_0| のように小文字を含む
;;;;   名前は保つ（Flax の flatten_dict が作る "params.Dense_0.kernel" と同じ名前にできる）。
;;;;   defmodule 以外で defpytree した型の子には名前が無いので、その節の下の葉の通し番号を
;;;;   使う（"state.0"、"state.1"）。木のルートが葉そのものなら名前は ""。
;;;; - 木全体が1つの名前空間なので、モデルとオプティマイザ状態は (defmodule train-state
;;;;   (params opt)) のように1つの木にまとめれば同じファイルに接頭辞つきで入る
;;;;   ("params.w"、"opt.0.mu")。別ファイルにしたければ別々に保存する。API に接頭辞の引数は持たない。
;;;; - 衝突しうる名前（スロット名に "." を含む |a.b| と、スロット a の子 b など）は保存・復元の
;;;;   前に :duplicate-path で報告する。
;;;; - 保存できる葉は配列と device-array（TO-HOST でホストに戻して書く。device-array かどうかは
;;;;   core が基底クラスを持たないので、DEVICE-ARRAY-AVAL のメソッドがあるかで見分ける）。
;;;;   (unsigned-byte 16) の配列は bf16 / f16 が決められないので、保存では :dtypes
;;;;   （パス名 → dtype の連想リスト）で言うか、device-array の aval から決める。
;;;; - 復元はファイルを読み、ひな形（init で作ったモデルなど）の葉と名前で対応づけて
;;;;   TREE-UNFLATTEN する。ひな形の葉は配列・device-array・aval のどれでもよい（値は使わず
;;;;   形と dtype だけを見る）。復元した葉はホストの新しい配列。
;;;; - 厳密に照合する: ファイルとひな形の名前の集合が違えば、足りなくても余っていても
;;;;   エラー。メタデータの "nabla." で始まるキーは予約（版の記録）。ユーザーのメタデータは
;;;;   そのまま往復するので、データローダの再開状態などを文字列にして入れられる。

(in-package #:nabla.nn)

(defparameter +checkpoint-version+ "1")
(defparameter +version-key+ "nabla.checkpoint_version")

(define-condition checkpoint-error (error)
  ((reason :initarg :reason :reader checkpoint-error-reason
           :documentation "原因を表すキーワード。")
   (name :initarg :name :initform nil :reader checkpoint-error-name
         :documentation "原因になった葉のパス名。葉が決まらないときは NIL。")
   (detail :initarg :detail :reader checkpoint-error-detail
           :documentation "人間向けの詳しい説明（文字列）。"))
  (:report (lambda (c s)
             (format s "checkpoint~@[ (~A)~]: ~A (~S)"
                     (checkpoint-error-name c) (checkpoint-error-detail c)
                     (checkpoint-error-reason c))))
  (:documentation
   "SAVE-CHECKPOINT / LOAD-CHECKPOINT が、木・ひな形・ファイルの組み合わせを扱えないときに
signal する。REASON は次のどれか。
  :inconsistent-tree   パス名の順序が TREE-FLATTEN と合わない（defpytree の登録が途中で変わったとき）
  :bad-leaf            配列でも device-array でも（ひな形では aval でも）ない葉
  :duplicate-path      2つの葉が同じパス名になる
  :unknown-path        :dtypes のパス名が木に無い
  :reserved-metadata   メタデータのキーが \"nabla.\" で始まる
  :unsupported-version ファイルの版がこの nabla の知らないもの
  :missing-entries     ひな形にある葉がファイルに無い
  :unexpected-entries  ファイルにある葉がひな形に無い
  :shape-mismatch      ファイルとひな形で葉の形が違う
  :dtype-mismatch      ファイルとひな形で葉の dtype が違う
NAME は :shape-mismatch / :dtype-mismatch / :bad-leaf / :duplicate-path / :unknown-path の
原因になった葉（のパス名）。ファイルが壊れているときは SAFETENSORS-ERROR になる。"))

(defun %checkpoint-fail (reason name control &rest args)
  (error 'checkpoint-error :reason reason :name name :detail (apply #'format nil control args)))

;;; --- パス名 ---

(defun %segment (symbol)
  (let ((name (symbol-name symbol)))
    (if (string= name (string-upcase name)) (string-downcase name) name)))

(defun %join (prefix segment)
  (if prefix (concatenate 'string prefix "." segment) segment))

(defun %named-leaves (tree prefix)
  "TREE の (パス名 . 葉) のリストを、NABLA:TREE-LEAVES と同じ順で返す。"
  (flet ((indexed (children)
           (loop for child in children
                 for i from 0
                 append (%named-leaves child (%join prefix (princ-to-string i))))))
    (cond ((listp tree) (indexed tree))
          ((simple-vector-p tree) (indexed (coerce tree 'list)))
          ((and (typep tree 'structure-object) (module-slots tree))
           (loop for (slot . kind) in (module-slots tree)
                 unless (eq kind :static)
                   append (%named-leaves (slot-value tree slot) (%join prefix (%segment slot)))))
          (t (let ((leaves (nabla:tree-leaves tree)))
               (if (and leaves (null (cdr leaves)) (eq (first leaves) tree))
                   (list (cons (or prefix "") tree))
                   (loop for leaf in leaves
                         for i from 0
                         collect (cons (%join prefix (princ-to-string i)) leaf))))))))

(defun %paths-and-leaves (tree)
  "(values パス名のリスト 葉のリスト treedef)。TREE-FLATTEN は1回だけ呼ぶ。"
  (multiple-value-bind (leaves treedef) (nabla:tree-flatten tree)
    (let ((named (%named-leaves tree nil)))
      (unless (and (= (length named) (length leaves)) (every #'eq leaves (mapcar #'cdr named)))
        (%checkpoint-fail :inconsistent-tree nil
                          "パス名の順序が TREE-FLATTEN と合わない（defpytree の登録が途中で変わった？）"))
      (values (mapcar #'car named) leaves treedef))))

(defun tree-paths (tree)
  "TREE の葉のパス名の文字列を、NABLA:TREE-LEAVES と同じ順に並べたリストで返す。
リストとベクタの子は添字、defmodule のモジュールの子はスロット名（:static は出ない。
Lisp の既定で大文字になる名前は小文字に、|Dense_0| のように小文字を含む名前はそのまま）、
defmodule 以外で defpytree した型の子は、その節の下の葉の通し番号。\".\" でつなぐ
（\"enc.layers.0.w\"）。木のルートが葉なら \"\"。名前の重複は検査しない
\(SAVE-CHECKPOINT / LOAD-CHECKPOINT が :duplicate-path にする)。"
  (values (%paths-and-leaves tree)))

(defun %check-paths (paths dtypes)
  "PATHS から作った名前の集合（equal のハッシュ表）を返す。重複と DTYPES の未知の名前は報告する。"
  (let ((seen (make-hash-table :test 'equal)))
    (dolist (path paths)
      (when (gethash path seen)
        (%checkpoint-fail :duplicate-path path "パス名 ~S が複数の葉に付く" path))
      (setf (gethash path seen) t))
    (dolist (entry dtypes)
      (unless (gethash (car entry) seen)
        (%checkpoint-fail :unknown-path (car entry) ":dtypes のパス名 ~S が木に無い" (car entry))))
    seen))

(defun %array-leaf-p (object)
  "文字列は配列だが数値のテンソルではないので葉として受けない（:bad-leaf）。"
  (and (arrayp object) (not (stringp object))))

(defun %device-array-p (object)
  (and (not (arrayp object))
       (compute-applicable-methods #'nabla:device-array-aval (list object))
       t))

(defun %dtype-for (path dtypes)
  (cdr (assoc path dtypes :test #'equal)))

;;; --- 保存 ---

(defun save-checkpoint (path tree &key metadata dtypes)
  "TREE の葉をパス名（TREE-PATHS）で safetensors の PATH に保存し、書いたパス名を返す。
葉は配列か device-array（TO-HOST でホストに戻す。dtype は aval から）。
\(unsigned-byte 16) の配列は DTYPES（(パス名 . :bf16 か :f16) のリスト）で dtype を言う
\(言わなければ NABLA:DTYPE-MISMATCH）。モデルとオプティマイザ状態を同じファイルに
入れるには、両方を1つの木（リストや defmodule）にまとめる。
METADATA は (キー . 値)（文字列）のリストで、そのまま保存して LOAD-CHECKPOINT の2値目に
返る。\"nabla.\" で始まるキーは予約（:reserved-metadata）。書き込みは
WRITE-SAFETENSORS のとおり一時ファイル経由で、失敗したら既存のファイルを壊さない。
不正な葉・名前の重複・DTYPES の未知のパス名は CHECKPOINT-ERROR。"
  (multiple-value-bind (paths leaves) (%paths-and-leaves tree)
    (%check-paths paths dtypes)
    (dolist (item metadata)
      (when (and (consp item) (stringp (car item)) (eql 0 (search "nabla." (car item))))
        (%checkpoint-fail :reserved-metadata nil "メタデータのキー ~S は予約されている" (car item))))
    (write-safetensors
     path
     (loop for name in paths
           for leaf in leaves
           collect (cond ((%array-leaf-p leaf)
                          (list name leaf (nabla:array-dtype leaf (%dtype-for name dtypes))))
                         ((%device-array-p leaf)
                          (list name (nabla:to-host leaf)
                                (or (%dtype-for name dtypes)
                                    (nabla:aval-dtype (nabla:device-array-aval leaf)))))
                         (t (%checkpoint-fail :bad-leaf name "葉 ~S は配列でも device-array でもない" leaf))))
     :metadata (append metadata (list (cons +version-key+ +checkpoint-version+))))))

;;; --- 復元 ---

(defun %template-shape-and-dtype (name leaf dtypes)
  "ひな形の葉 LEAF の (values 形 dtype) を返す。dtype が決められない（dtype を言っていない
\(unsigned-byte 16) の配列）ときは NIL。"
  (let ((dtype (%dtype-for name dtypes)))
    (cond ((%array-leaf-p leaf)
           (values (array-dimensions leaf)
                   (if (or dtype (not (equal (array-element-type leaf) '(unsigned-byte 16))))
                       (handler-case (nabla:array-dtype leaf dtype)
                         (nabla:dtype-mismatch (c)
                           (%checkpoint-fail :dtype-mismatch name "ひな形の葉の要素型 ~S は dtype ~S と合わない"
                                             (nabla:dtype-mismatch-element-type c) dtype)))
                       nil)))
          ((%device-array-p leaf)
           (let ((aval (nabla:device-array-aval leaf)))
             (values (nabla:aval-shape aval) (or dtype (nabla:aval-dtype aval)))))
          ((nabla:aval-p leaf)
           (values (nabla:aval-shape leaf) (or dtype (nabla:aval-dtype leaf))))
          (t (%checkpoint-fail :bad-leaf name "ひな形の葉 ~S は配列でも device-array でも aval でもない" leaf)))))

(defun load-checkpoint (path template &key dtypes)
  "PATH の safetensors を、TEMPLATE と同じ構造の木に復元する。(values 木 メタデータ) を返す。
TEMPLATE は init で作ったモデルなど、保存した木と同じパス名（TREE-PATHS）を持つ木で、
葉は配列・device-array・aval のどれでもよい（形と dtype だけを使う）。復元した葉はホストの
新しい配列で、木の構造（treedef）は TEMPLATE のもの。メタデータは保存時の METADATA
\(\"nabla.\" で始まる予約のキーを除く)。
ファイルとひな形の名前が一致しない（:missing-entries / :unexpected-entries）、形が違う
\(:shape-mismatch)、dtype が違う（:dtype-mismatch）、版が新しい（:unsupported-version）ときは
CHECKPOINT-ERROR。dtype を言っていない (unsigned-byte 16) のひな形の葉は bf16 / f16 の
どちらのファイルでも受けるので、区別したければ DTYPES（SAVE-CHECKPOINT と同じ）で言う。
壊れたファイルは SAFETENSORS-ERROR。"
  (multiple-value-bind (paths leaves treedef) (%paths-and-leaves template)
    (let ((names (%check-paths paths dtypes)))
      (let ((expected (loop for name in paths
                            for leaf in leaves
                            collect (multiple-value-bind (shape dtype)
                                        (%template-shape-and-dtype name leaf dtypes)
                                      (list name shape dtype)))))
        (multiple-value-bind (entries metadata) (read-safetensors path)
          (let ((version (cdr (assoc +version-key+ metadata :test #'string=)))
                (table (make-hash-table :test 'equal)))
            (when (and version (string/= version +checkpoint-version+))
              (%checkpoint-fail :unsupported-version nil "ファイルの版 ~S は未対応（この nabla は ~S）"
                                version +checkpoint-version+))
            (dolist (entry entries) (setf (gethash (first entry) table) entry))
            (let ((missing (remove-if (lambda (name) (gethash name table)) paths))
                  (unexpected (remove-if (lambda (name) (gethash name names)) (mapcar #'first entries))))
              (when missing
                (%checkpoint-fail :missing-entries nil "ファイルに無い葉: ~{~S~^ ~}" missing))
              (when unexpected
                (%checkpoint-fail :unexpected-entries nil "ひな形に無い葉: ~{~S~^ ~}" unexpected)))
            (values
             (nabla:tree-unflatten
              treedef
              (loop for (name shape dtype) in expected
                    collect (destructuring-bind (array file-dtype) (rest (gethash name table))
                              (unless (equal shape (array-dimensions array))
                                (%checkpoint-fail :shape-mismatch name "形がひな形 ~S とファイル ~S で違う"
                                                  shape (array-dimensions array)))
                              (unless (if dtype (eq dtype file-dtype) (member file-dtype '(:bf16 :f16)))
                                (%checkpoint-fail :dtype-mismatch name "dtype がひな形 ~S とファイル ~S で違う"
                                                  (or dtype :bf16/f16) file-dtype))
                              array)))
             (remove-if (lambda (item) (eql 0 (search "nabla." (car item)))) metadata))))))))
