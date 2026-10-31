;;;; PyTree のチェックポイント（issue #206）。
;;;;
;;;; 守らせる性質:
;;;; - パス名（small）: 葉と同じ個数で、重複せず、構造だけで決まる（葉の値を変えても同じ）。
;;;;   リスト・ベクタは添字、defmodule のモジュールはスロット名（:static は葉でないので出ない）、
;;;;   defpytree の FLATTEN が子の名前を返す型はその名前、返さない型は添字。
;;;; - 往復（medium）: save → load は、全 dtype（bf16 / f16 / i1 を含む）の葉をビット列まで保ち、
;;;;   treedef とメタデータも保つ。
;;;; - ひな形と合わないとき（medium）: 名前の過不足・形・dtype の違いを、理由つきの
;;;;   CHECKPOINT-ERROR で報告する。
;;;; - 相互運用（medium）: Flax の flatten_dict と同じパス名で Python の safetensors が書いた
;;;;   ファイルを復元し、numpy と同じ順伝播の結果になる。

(in-package #:nabla.nn.tests)

(defparameter *ck-dtypes* '(:f32 :f64 :bf16 :f16 :i1 :i32 :u32 :u64))

;;; テスト用のモジュールと、defmodule 以外で登録する節
(defmodule ck-layer ((w :param) (b :state) (act :static)))
(defmodule ck-model ((enc :param) (layers :param) (aux :param) (tag :static)))
(defmodule ck-enc ((layers :param) (|Dense_0| :param) (stats :state) (width :static)))
(defmodule ck-inner ((b :param)))
(defmodule ck-ambiguous ((|a.b| :param) (a :param)))
(defmodule ck-train-state ((params :param) (opt :param)))
(defmodule ck-dense ((kernel :param) (bias :param)))
(defmodule ck-mlp ((|Dense_0| :param) (|Dense_1| :param)))
(defmodule ck-variables ((params :param)))

(defstruct ck-pair a b)
(nabla:defpytree ck-pair
  :flatten (lambda (p) (values (list (ck-pair-a p) (ck-pair-b p)) nil))
  :unflatten (lambda (aux children) (declare (ignore aux))
               (make-ck-pair :a (first children) :b (second children))))

;;; FLATTEN が3つ目の値で子の名前を返す節（nabla/data のローダの再開状態の形。#285）
(defstruct ck-loader key epoch position)
(nabla:defpytree ck-loader
  :flatten (lambda (s) (values (list (ck-loader-key s) (ck-loader-epoch s) (ck-loader-position s))
                               nil '("key" "epoch" "position")))
  :unflatten (lambda (aux children) (declare (ignore aux))
               (destructuring-bind (key epoch position) children
                 (make-ck-loader :key key :epoch epoch :position position))))

;;; device-array の代わり（to-host と device-array-aval のメソッドだけを持つ）
(defclass fake-device-array ()
  ((host :initarg :host) (dtype :initarg :dtype)))

(defun make-fake-device-array (host dtype)
  (make-instance 'fake-device-array :host host :dtype dtype))

(defmethod nabla:to-host ((array fake-device-array))
  (copy-seq (slot-value array 'host)))

(defmethod nabla:device-array-aval ((array fake-device-array))
  (nabla:make-aval (array-dimensions (slot-value array 'host)) (slot-value array 'dtype)))

;;; --- 生成 ---

(defun ck-random-array (state)
  (random-array-of (nth (random (length *ck-dtypes*) state) *ck-dtypes*)
                   (loop repeat (random 4 state) collect (random 4 state))
                   state))

(defun ck-random-tree (state depth)
  "STATE から決まる、リスト・ベクタ・ck-layer・ck-model・ck-pair・ck-loader・nil・配列の入れ子の木。"
  (flet ((sub () (ck-random-tree state (1- depth))))
    (if (<= depth 0)
        (ck-random-array state)
        (ecase (random 9 state)
          ((0 7) (ck-random-array state))
          (1 (loop repeat (random 4 state) collect (sub)))
          (2 (coerce (loop repeat (random 4 state) collect (sub)) 'simple-vector))
          (3 (make-ck-layer :w (sub) :b (sub) :act :relu))
          (4 (make-ck-model :enc (sub) :layers (loop repeat (random 3 state) collect (sub))
                            :aux (sub) :tag (list :n (random 3 state))))
          (5 (make-ck-pair :a (sub) :b (sub)))
          (6 nil)
          (8 (make-ck-loader :key (sub) :epoch (sub) :position (sub)))))))

(defun ck-dtype-map (tree seed)
  "TREE の (unsigned-byte 16) の葉ごとに bf16 / f16 を SEED から選んだ (パス . dtype) のリスト。"
  (let ((state (sb-ext:seed-random-state seed)))
    (loop for path in (tree-paths tree)
          for leaf in (nabla:tree-leaves tree)
          when (equal (array-element-type leaf) '(unsigned-byte 16))
            collect (cons path (if (zerop (random 2 state)) :bf16 :f16)))))

(defun ck-same-leaves-p (a b)
  (let ((la (nabla:tree-leaves a)) (lb (nabla:tree-leaves b)))
    (and (= (length la) (length lb)) (every #'same-array-p la lb))))

(defun ck-f32 (&rest values)
  (make-array (length values) :element-type 'single-float :initial-contents values))

(defun ck-error-reason (thunk)
  (handler-case (progn (funcall thunk) :no-error)
    (checkpoint-error (c) (checkpoint-error-reason c))))

(defmacro def-ck-property (name suite docstring (&rest vars) &body body)
  "(seed depth) を VARS に束縛して BODY（真偽）を PBT で確かめる。"
  `(test (,name :suite ,suite)
     ,docstring
     (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1000000)
                                     (uniform-integer :lo 1 :hi 4)))
                   (lambda (case) (destructuring-bind ,vars case ,@body))
                   :regression-id ,name
                   :regression-file (regression-path ,(substitute #\- #\/ (string-downcase (string name)))
                                                     :package "NABLA.NN.TESTS"))
         ,(format nil "~A の性質が成り立たなかった" name))))

(defmacro with-checkpoint-file ((path) &body body)
  `(with-temporary-directory (dir)
     (let ((,path (merge-pathnames "ckpt.safetensors" dir)))
       ,@body)))

;;; --- パス名（small） ---

(def-ck-property checkpoint/paths-are-unique-and-one-per-leaf :nabla.small
  "パス名は葉と同じ個数で、重複しない。"
  (seed depth)
  (let* ((tree (ck-random-tree (sb-ext:seed-random-state seed) depth))
         (paths (tree-paths tree)))
    (and (= (length paths) (nabla:treedef-num-leaves (nabla:tree-structure tree)))
         (= (length paths) (length (remove-duplicates paths :test #'string=))))))

(def-ck-property checkpoint/paths-depend-only-on-structure :nabla.small
  "葉の値を変えても（tree-map）パス名は同じ。"
  (seed depth)
  (let ((tree (ck-random-tree (sb-ext:seed-random-state seed) depth)))
    (equal (tree-paths tree)
           (tree-paths (nabla:tree-map (lambda (leaf)
                                         (make-array (array-dimensions leaf)
                                                     :element-type (array-element-type leaf)))
                                       tree)))))

(test (checkpoint/paths-use-indices-and-slot-names :suite :nabla.small)
  (flet ((leaf () (ck-f32 0.0)))
    (let ((model (make-ck-model
                  :enc (make-ck-enc :layers (list (make-ck-layer :w (leaf) :b (leaf) :act :relu)
                                                  (make-ck-layer :w (leaf) :b (leaf) :act :relu))
                                    :|Dense_0| (leaf) :stats (vector (leaf) nil (leaf)) :width 3)
                  :layers nil :aux (list (list (leaf))) :tag 1)))
      ;; スロット名は小文字に、|Dense_0| のような大文字小文字の混じる名前はそのまま。
      ;; :static は出ず、nil は葉を持たず、ベクタの添字は nil の分も数える
      (is (equal '("enc.layers.0.w" "enc.layers.0.b" "enc.layers.1.w" "enc.layers.1.b"
                   "enc.Dense_0" "enc.stats.0" "enc.stats.2" "aux.0.0")
                 (tree-paths model))))))

(test (checkpoint/paths-of-bare-leaves-and-registered-nodes :suite :nabla.small)
  (let ((leaf (make-array 0 :element-type 'single-float)))
    (is (equal '("") (tree-paths leaf)))
    (is (equal '() (tree-paths nil)))
    ;; defmodule 以外の登録した型の子には、FLATTEN が返す子の添字を使い、入れ子は再帰する
    (is (equal '("0" "1.0" "1.1.0") (tree-paths (list leaf (make-ck-pair :a leaf :b (list leaf))))))))

(defstruct ck-adam mu nu count)
(nabla:defpytree ck-adam
  :flatten (lambda (a) (values (list (ck-adam-mu a) (ck-adam-nu a) (ck-adam-count a)) nil))
  :unflatten (lambda (aux children) (declare (ignore aux))
               (make-ck-adam :mu (first children) :nu (second children) :count (third children))))

(test (checkpoint/registered-node-keeps-nested-module-names :suite :nabla.small)
  "オプティマイザ状態のような defpytree の節が defmodule のサブ木を持つとき、w / b の名前が残る。"
  (flet ((layer () (make-ck-layer :w (ck-f32 0.0) :b (ck-f32 0.0) :act :relu)))
    (let ((state (make-ck-train-state
                  :params (layer)
                  :opt (make-ck-adam :mu (layer) :nu (list (layer)) :count (ck-f32 0.0)))))
      (is (equal '("params.w" "params.b" "opt.0.w" "opt.0.b" "opt.1.0.w" "opt.1.0.b" "opt.2")
                 (tree-paths state))))))

(test (checkpoint/registered-node-with-child-names-uses-the-names :suite :nabla.small)
  "defpytree の FLATTEN が子の名前を返す節の子は、添字でなくその名前で \"親.名前\" になり、
入れ子もたどる。名前を返さない ck-pair の子は今までどおり添字のまま。"
  (let ((leaf (ck-f32 0.0)))
    (is (equal '("params.w" "params.b" "opt.0.key" "opt.0.epoch" "opt.0.position.0" "opt.0.position.1"
                 "opt.1.0" "opt.1.1")
               (tree-paths (make-ck-train-state
                            :params (make-ck-layer :w leaf :b leaf :act :relu)
                            :opt (list (make-ck-loader :key leaf :epoch leaf :position (list leaf leaf))
                                       (make-ck-pair :a leaf :b leaf))))))))

(defmodule ck-empty ((|| :param)))

(test (checkpoint/empty-slot-names-are-rejected :suite :nabla.small)
  (is (eq :empty-segment (ck-error-reason (lambda () (tree-paths (make-ck-empty :|| (ck-f32 1.0))))))))

;;; --- 往復と不一致（medium） ---

(def-ck-property checkpoint/save-then-load-is-bit-exact :nabla.medium
  "save → load は、全 dtype（bf16 / f16 / i1 を含む）の葉をビット列まで、treedef とメタデータごと保つ。"
  (seed depth)
  (let* ((tree (ck-random-tree (sb-ext:seed-random-state seed) depth))
         (dtypes (ck-dtype-map tree seed))
         (metadata (random-metadata seed)))
    (with-checkpoint-file (path)
      (save-checkpoint path tree :dtypes dtypes :metadata metadata)
      (multiple-value-bind (back loaded-metadata) (load-checkpoint path tree :dtypes dtypes)
        (and (nabla:treedef= (nabla:tree-structure tree) (nabla:tree-structure back))
             (ck-same-leaves-p tree back)
             (equal (sort (copy-list metadata) #'string< :key #'car)
                    (sort (copy-list loaded-metadata) #'string< :key #'car)))))))

;;; --- データパイプラインの状態（issue #244） ---
;;; エポック数・バッチ内の位置は rank 0 の :i32 配列、シャッフルの PRNG キーは prng-key が作る
;;; :u32 (2) の配列、小さな配列（バッファの先頭など）。Lisp の整数は葉にできないので配列で持つ。

;;; nabla/data は nn に依存できないので、状態は defmodule でなく子に名前を付けた defpytree の型にする（#285）
(defstruct ck-pipeline epoch position shuffle-key buffer)
(nabla:defpytree ck-pipeline
  :flatten (lambda (p) (values (list (ck-pipeline-epoch p) (ck-pipeline-position p)
                                     (ck-pipeline-shuffle-key p) (ck-pipeline-buffer p))
                               nil '("epoch" "position" "shuffle-key" "buffer")))
  :unflatten (lambda (aux children) (declare (ignore aux))
               (destructuring-bind (epoch position shuffle-key buffer) children
                 (make-ck-pipeline :epoch epoch :position position :shuffle-key shuffle-key
                                   :buffer buffer))))

(defun ck-i32-scalar (value)
  (make-array nil :element-type '(signed-byte 32) :initial-element value))

(defun ck-random-pipeline (state)
  "STATE から決まるパイプラインの状態。整数は符号付き32ビットの端（最小・最大）も取り、
キーは符号付き・符号なし64ビットのシードから作る。"
  (flet ((i32 ()
           (ck-i32-scalar (ecase (random 4 state)
                            (0 (- (expt 2 31))) (1 (1- (expt 2 31)))
                            ((2 3) (- (random (expt 2 32) state) (expt 2 31)))))))
    (make-ck-pipeline
     :epoch (i32) :position (i32)
     :shuffle-key (nabla:prng-key (- (random (expt 2 64) state) (random (expt 2 63) state)))
     :buffer (random-array-of (nth (random 3 state) '(:f32 :i32 :u32)) (list (random 5 state)) state))))

(def-ck-property checkpoint/pipeline-state-in-the-train-state-tree-round-trips :nabla.medium
  "モデル + オプティマイザ状態 + データパイプラインの状態(整数・PRNG キー・小さな配列)を1つの木に入れて
save → load すると、全部がビット列まで戻り、復元したキーは元のキーと同じ split の結果を作る。"
  (seed depth)
  (let* ((state (sb-ext:seed-random-state seed))
         (model (ck-random-tree state depth))
         (pipeline (ck-random-pipeline state))
         (tree (make-ck-train-state :params model :opt (list (ck-random-tree state depth) pipeline)))
         (dtypes (ck-dtype-map tree seed)))
    (with-checkpoint-file (path)
      (save-checkpoint path tree :dtypes dtypes)
      (let* ((back (load-checkpoint path tree :dtypes dtypes))
             (back-pipeline (second (ck-train-state-opt back))))
        (and (nabla:treedef= (nabla:tree-structure tree) (nabla:tree-structure back))
             (ck-same-leaves-p tree back)
             (same-array-p (ck-pipeline-epoch pipeline) (ck-pipeline-epoch back-pipeline))
             (same-array-p (ck-pipeline-shuffle-key pipeline) (ck-pipeline-shuffle-key back-pipeline))
             (eq :u32 (nabla:array-dtype (ck-pipeline-shuffle-key back-pipeline)))
             (equal '(2) (array-dimensions (ck-pipeline-shuffle-key back-pipeline)))
             (equal nil (array-dimensions (ck-pipeline-position back-pipeline)))
             (same-array-p (nabla:split (ck-pipeline-shuffle-key pipeline) 3)
                           (nabla:split (ck-pipeline-shuffle-key back-pipeline) 3)))))))

(test (checkpoint/pipeline-state-paths-are-slot-names :suite :nabla.small)
  (is (equal '("epoch" "position" "shuffle-key" "buffer")
             (tree-paths (ck-random-pipeline (sb-ext:seed-random-state 1))))))

(test (checkpoint/named-registered-node-round-trips-under-its-names :suite :nabla.medium)
  "子の名前を返す defpytree の節（ローダの再開状態の形）は、ファイルの中で \"親.名前\" のエントリに
なり、save → load でビット列まで往復する。"
  (let ((tree (make-ck-train-state
               :params (make-ck-layer :w (ck-f32 1.0 2.0) :b (ck-f32 3.0) :act :relu)
               :opt (make-ck-loader :key (nabla:prng-key 7) :epoch (ck-i32-scalar 3)
                                    :position (ck-i32-scalar -5)))))
    (with-checkpoint-file (path)
      (save-checkpoint path tree)
      (is (equal '("opt.epoch" "opt.key" "opt.position" "params.b" "params.w")
                 (sort (mapcar #'first (read-safetensors path)) #'string<)))
      (let ((back (load-checkpoint path tree)))
        (is (typep (ck-train-state-opt back) 'ck-loader))
        (is (nabla:treedef= (nabla:tree-structure tree) (nabla:tree-structure back)))
        (is (ck-same-leaves-p tree back))))))

(test (checkpoint/lisp-integers-are-not-leaves :suite :nabla.medium)
  "データパイプラインの状態は配列で持つ。Lisp の整数を葉にすると :bad-leaf（保存もひな形も）。"
  (with-checkpoint-file (path)
    (is (eq :bad-leaf (ck-error-reason
                       (lambda () (save-checkpoint path (make-ck-pipeline :epoch 3 :position (ck-i32-scalar 0)
                                                                          :shuffle-key (nabla:prng-key 0)
                                                                          :buffer (ck-f32)))))))
    (save-checkpoint path (ck-random-pipeline (sb-ext:seed-random-state 2)))
    (is (eq :bad-leaf (ck-error-reason
                       (lambda () (load-checkpoint path (make-ck-pipeline :epoch 3 :position 0
                                                                          :shuffle-key 0 :buffer 0))))))))

(defun ck-perturb-leaf (tree index kind)
  "TREE の INDEX 番目の葉を、KIND（:shape か :dtype）でひな形として食い違うものにする。"
  (let ((i -1))
    (nabla:tree-map
     (lambda (leaf)
       (if (/= (incf i) index)
           leaf
           (ecase kind
             (:shape (make-array (if (array-dimensions leaf)
                                     (cons (1+ (first (array-dimensions leaf)))
                                           (rest (array-dimensions leaf)))
                                     '(1))
                                 :element-type (array-element-type leaf)))
             (:dtype (make-array (array-dimensions leaf)
                                 :element-type (if (eq 'single-float (array-element-type leaf))
                                                   'double-float
                                                   'single-float))))))
     tree)))

(def-ck-property checkpoint/mismatched-template-is-reported :nabla.medium
  "ひな形の1つの葉の形か dtype を変えると、その葉のパス名つきの CHECKPOINT-ERROR になる。
葉を足すと :missing-entries、減らすと :unexpected-entries。"
  (seed depth)
  (let* ((tree (list (ck-random-tree (sb-ext:seed-random-state seed) depth)))
         (n (length (nabla:tree-leaves tree))))
    (with-checkpoint-file (path)
      (save-checkpoint path tree :dtypes (ck-dtype-map tree seed))
      (flet ((outcome (template &optional (dtypes (ck-dtype-map tree seed)))
               (handler-case (progn (load-checkpoint path template :dtypes dtypes) :loaded)
                 (checkpoint-error (c) (list (checkpoint-error-reason c) (checkpoint-error-name c))))))
        (and (equal '(:missing-entries nil)
                    (outcome (list (first tree) (vector (make-array 0 :element-type 'single-float)))))
             (or (zerop n)
                 (let* ((index (mod seed n))
                        (name (nth index (tree-paths tree))))
                   (and (equal '(:unexpected-entries nil) (outcome (list nil) nil))
                        (equal (list :shape-mismatch name) (outcome (ck-perturb-leaf tree index :shape)))
                        (equal (list :dtype-mismatch name) (outcome (ck-perturb-leaf tree index :dtype)))))))))))

;;; --- 例ベース（medium） ---

(test (checkpoint/dtypes-decide-bf16-and-f16 :suite :nabla.medium)
  (with-checkpoint-file (path)
    (let ((tree (list (make-array 2 :element-type '(unsigned-byte 16) :initial-contents '(#x3F80 #xBF80)))))
      ;; dtype を言わない (unsigned-byte 16) の葉は保存できない
      (signals nabla:dtype-mismatch (save-checkpoint path tree))
      (is (eq :unknown-path (ck-error-reason (lambda () (save-checkpoint path tree :dtypes '(("9" . :bf16)))))))
      (save-checkpoint path tree :dtypes '(("0" . :bf16)))
      (is (eq :bf16 (third (find "0" (read-safetensors path) :key #'first :test #'string=))))
      ;; ひな形の (unsigned-byte 16) の葉も保存と同じく dtype を言う（言わなければ生のビットを
      ;; bf16 / f16 のどちらのファイルからでも受けてしまう）。言えば file の dtype と比べる
      (is (eq :dtype-mismatch (ck-error-reason (lambda () (load-checkpoint path tree)))))
      (is (ck-same-leaves-p tree (load-checkpoint path tree :dtypes '(("0" . :bf16)))))
      (is (eq :dtype-mismatch
              (ck-error-reason (lambda () (load-checkpoint path tree :dtypes '(("0" . :f16)))))))
      ;; 要素型と合わない dtype を言ったひな形も CHECKPOINT-ERROR
      (is (eq :dtype-mismatch
              (ck-error-reason (lambda () (load-checkpoint path (list (ck-f32 1.0 2.0))
                                                           :dtypes '(("0" . :bf16))))))))))

(test (checkpoint/device-arrays-are-saved-from-the-host-and-avals-are-templates :suite :nabla.medium)
  (with-checkpoint-file (path)
    (let* ((host (make-array 2 :element-type '(unsigned-byte 16) :initial-contents '(#x3C00 #x4000)))
           (tree (list (make-fake-device-array host :f16) (ck-f32 1.0 2.0))))
      ;; dtype は device-array の aval から決まる
      (save-checkpoint path tree)
      (is (eq :f16 (third (find "0" (read-safetensors path) :key #'first :test #'string=))))
      (let ((back (load-checkpoint path (list (nabla:make-aval '(2) :f16) (nabla:make-aval '(2) :f32)))))
        (is (equalp host (first back)))
        (is (equalp (ck-f32 1.0 2.0) (second back))))
      ;; device-array のひな形: 復元されるのはホストの配列
      (is (arrayp (first (load-checkpoint path tree))))
      (is (eq :dtype-mismatch
              (ck-error-reason (lambda () (load-checkpoint path (list (nabla:make-aval '(2) :f32)
                                                                      (nabla:make-aval '(2) :f32))))))))))

(test (checkpoint/rejects-bad-leaves-and-reserved-metadata :suite :nabla.medium)
  (with-checkpoint-file (path)
    (is (eq :bad-leaf (ck-error-reason (lambda () (save-checkpoint path (list 1.0))))))
    (is (eq :bad-leaf (ck-error-reason (lambda () (save-checkpoint path (list "text"))))))
    (is (eq :reserved-metadata
            (ck-error-reason (lambda () (save-checkpoint path (list (ck-f32 1.0))
                                                         :metadata '(("nabla.x" . "1")))))))
    (is (not (probe-file path)) "失敗した保存はファイルを作らない")
    (save-checkpoint path (list (ck-f32 1.0)))
    (is (eq :bad-leaf (ck-error-reason (lambda () (load-checkpoint path (list 1.0))))))))

(test (checkpoint/duplicate-paths-are-an-error :suite :nabla.medium)
  "スロット名に \".\" を含む |a.b| と、スロット a の子 b は同じパス名 \"a.b\" になる。"
  (with-checkpoint-file (path)
    (let ((clash (make-ck-ambiguous :|a.b| (ck-f32 1.0) :a (make-ck-inner :b (ck-f32 2.0)))))
      (is (equal '("a.b" "a.b") (tree-paths clash)))
      (is (eq :duplicate-path (ck-error-reason (lambda () (save-checkpoint path clash)))))
      (is (eq :duplicate-path (ck-error-reason (lambda () (load-checkpoint path clash)))))
      (is (not (probe-file path))))))

(defmodule ck-reordered ((x :param) (y :param)))

(test (checkpoint/path-order-must-match-tree-flatten :suite :nabla.small)
  "defmodule の登録を、子の順が違う FLATTEN で上書きすると、名前と葉の対応が崩れるので報告する。"
  (let ((tree (make-ck-reordered :x (ck-f32 1.0) :y (ck-f32 2.0))))
    (is (equal '("x" "y") (tree-paths tree)))
    (nabla:defpytree ck-reordered
      :flatten (lambda (m) (values (list (ck-reordered-y m) (ck-reordered-x m)) nil))
      :unflatten (lambda (aux children) (declare (ignore aux))
                   (make-ck-reordered :x (second children) :y (first children))))
    (is (eq :inconsistent-tree (ck-error-reason (lambda () (tree-paths tree)))))))

(test (checkpoint/version-and-metadata :suite :nabla.medium)
  (with-checkpoint-file (path)
    (let ((tree (list (ck-f32 1.0))))
      (save-checkpoint path tree :metadata '(("step" . "7")))
      (let ((metadata (nth-value 1 (read-safetensors path))))
        (is (equal "1" (cdr (assoc "nabla.checkpoint_version" metadata :test #'string=)))))
      (is (equal '(("step" . "7")) (nth-value 1 (load-checkpoint path tree))))
      ;; 将来の版のファイルは読まない
      (write-safetensors path (list (list "0" (ck-f32 1.0)))
                         :metadata '(("nabla.checkpoint_version" . "2")))
      (is (eq :unsupported-version (ck-error-reason (lambda () (load-checkpoint path tree)))))
      ;; 版の無い（Python が書いた）ファイルは読む
      (write-safetensors path (list (list "0" (ck-f32 1.0))))
      (is (ck-same-leaves-p tree (load-checkpoint path tree))))))

(test (checkpoint/saves-model-and-optimizer-state-in-one-file :suite :nabla.medium)
  (with-checkpoint-file (path)
    (let* ((model (make-ck-layer :w (ck-f32 1.0 2.0) :b (ck-f32 3.0) :act :relu))
           (opt (list (make-array '() :element-type '(unsigned-byte 32) :initial-element 5)
                      (list (ck-f32 0.5 0.5))))
           (state (make-ck-train-state :params model :opt opt)))
      (save-checkpoint path state)
      (is (equal '("params.w" "params.b" "opt.0" "opt.1.0") (tree-paths state)))
      (let ((back (load-checkpoint path state)))
        (is (typep back 'ck-train-state))
        (is (ck-same-leaves-p state back))
        (is (eq :relu (ck-layer-act (ck-train-state-params back))))))))

;;; --- Flax との相互運用（medium、例ベース） ---

(defun ck-affine (x kernel bias &key relu)
  "x (n, i) @ kernel (i, o) + bias (o)。float32 の素朴な3重ループ。"
  (let* ((n (array-dimension x 0)) (i (array-dimension kernel 0)) (o (array-dimension kernel 1))
         (y (make-array (list n o) :element-type 'single-float)))
    (dotimes (r n y)
      (dotimes (c o)
        (let ((sum (aref bias c)))
          (dotimes (k i) (incf sum (* (aref x r k) (aref kernel k c))))
          (setf (aref y r c) (if relu (max sum 0.0) sum)))))))

(test (checkpoint/restores-a-flax-written-file :suite :nabla.medium)
  "tests/fixtures/checkpoint/generate.py が Flax の flatten_dict と同じ規則のパス名で書いた
ファイルを復元し、numpy と同じ順伝播の結果になる（値は generate.py の出力）。"
  (flet ((zeros (&rest shape) (make-array shape :element-type 'single-float)))
    (let* ((template (make-ck-variables
                      :params (make-ck-mlp
                               :|Dense_0| (make-ck-dense :kernel (zeros 3 4) :bias (zeros 4))
                               :|Dense_1| (make-ck-dense :kernel (zeros 4 2) :bias (zeros 2)))))
           (x (make-array '(2 3) :element-type 'single-float
                                 :initial-contents '((1.0 -2.0 0.5) (0.0 1.0 3.0)))))
      (is (equal '("params.Dense_0.kernel" "params.Dense_0.bias"
                   "params.Dense_1.kernel" "params.Dense_1.bias")
                 (tree-paths template)))
      (multiple-value-bind (model metadata)
          (load-checkpoint (asdf:system-relative-pathname "nabla" "tests/fixtures/checkpoint/flax-mlp.safetensors")
                           template)
        (is (equal '(("format" . "flax")) metadata))
        (let* ((p (ck-variables-params model))
               (d0 (ck-mlp-|Dense_0| p)) (d1 (ck-mlp-|Dense_1| p))
               (y (ck-affine (ck-affine x (ck-dense-kernel d0) (ck-dense-bias d0) :relu t)
                             (ck-dense-kernel d1) (ck-dense-bias d1))))
          (is (equalp #2A((0.015625 -0.65625) (1.28125 2.375)) y)))))))

(defun run-checkpoint-tests ()
  "このファイルのテストだけを実行する（変更した行への mutation testing の --test-form 用）。"
  (let ((names (loop for symbol being the symbols of (find-package '#:nabla.nn.tests)
                     when (and (eq (symbol-package symbol) (find-package '#:nabla.nn.tests))
                               (eql 0 (search "CHECKPOINT/" (symbol-name symbol)))
                               (fiveam:get-test symbol))
                       collect symbol)))
    (fiveam:run! names)))

(test (nn/sources/do-not-reference-core-internals :suite :nabla.medium)
  "src/nn/ 以下の .lisp（再帰。checkpoint.lisp を含む1件以上）には、core の内部シンボルを
参照する NABLA:: と NB:: が一度も現れない（nn は core の公開 API だけを使う。issue #239。
コメントや docstring の言及も数える）。import-from #:nabla は禁止文字列に加えない
（nn は :use で公開シンボルだけを取り込むため、内部の取り込みは :: の形でしか書けない）。"
  (let ((offending nil)
        (paths (directory (merge-pathnames "**/*.lisp"
                                           (asdf:system-relative-pathname "nabla/nn" "src/nn/")))))
    (is (find "checkpoint" paths :key #'pathname-name :test #'string=))
    (dolist (path paths)
      (with-open-file (stream path :direction :input :external-format :utf-8)
        (let* ((text (make-string (file-length stream)))
               (text (string-downcase (subseq text 0 (read-sequence text stream)))))
          (when (or (search "nabla::" text) (search "nb::" text))
            (push path offending)))))
    (is (null offending) "core の内部シンボルを参照する nn のファイル: ~S" offending)))

(test (nn/tree-children-of-a-module-lists-non-static-slots-in-declaration-order :suite :nabla.small)
  "defmodule のインスタンスに tree-children を当てると、子は :param / :state のスロット値の
宣言順で、:static のスロットは含まない。節（node-p は真）。"
  (let* ((layer (make-ck-layer :w 1 :b 2 :act :relu)))
    (multiple-value-bind (children aux node-p) (nabla:tree-children layer)
      (declare (ignore aux))
      (is (equal '(1 2) children))
      (is (eq t node-p)))))
