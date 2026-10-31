;;;; map / filter の変換（issue #281）。
;;;;
;;;; 守らせる性質（small）:
;;;; - 合成: 変換のリストを各レコードに APPLY-TRANSFORMS でかけ、残ったものを順に集めた結果は、
;;;;   レコードのリストに先頭の変換から順に mapcar（map）/ remove-if-not（filter）をかけた結果と同じ。
;;;; - 捨てたレコード: filter が落としたら (values nil nil)、残ったら 2 つ目の値は t。
;;;; - 変換でないもの（batch など）を変換のリストに入れると type-error。

(in-package #:nabla.data.tests)

;;; --- 生成 ---

(defun transform-random-spec (state)
  "変換の仕様（:add k / :mul k / :keep-mod m r）を 1 つ作る。"
  (ecase (random 3 state)
    (0 (list :add (- (random 21 state) 10)))
    (1 (list :mul (- (random 7 state) 3)))
    (2 (let ((m (1+ (random 4 state)))) (list :keep-mod m (random m state))))))

(defun transform-spec-function (spec)
  "SPEC の関数（map ならレコード → レコード、filter なら述語）。"
  (destructuring-bind (kind a &optional b) spec
    (ecase kind
      (:add (lambda (x) (+ x a)))
      (:mul (lambda (x) (* x a)))
      (:keep-mod (lambda (x) (= b (mod x a)))))))

(defun spec-transform (spec)
  "SPEC を nabla.data の変換オブジェクトにする。"
  (if (eq (first spec) :keep-mod)
      (filter-transform (transform-spec-function spec))
      (map-transform (transform-spec-function spec))))

(defun list-oracle (specs records)
  "RECORDS に SPECS を先頭から順に、リスト全体への mapcar / remove-if-not としてかけた結果。"
  (reduce (lambda (list spec)
            (if (eq (first spec) :keep-mod)
                (remove-if-not (transform-spec-function spec) list)
                (mapcar (transform-spec-function spec) list)))
          specs :initial-value records))

(defun transformed-records (transforms records)
  "RECORDS の各レコードに APPLY-TRANSFORMS をかけ、残ったものを順に集めたリスト。"
  (loop for record in records
        for (out keep) = (multiple-value-list (apply-transforms transforms record))
        when keep collect out))

(defun transform-random-case (seed)
  "SEED から決まる (values 変換の仕様のリスト レコード（整数）のリスト)。"
  (let ((state (sb-ext:seed-random-state seed)))
    (values (loop repeat (random 6 state) collect (transform-random-spec state))
            (loop repeat (random 8 state) collect (- (random 41 state) 20)))))

;;; --- 合成 ---

(def-source-property transforms/compose-like-mapcar-and-remove-if-not
  "レコードごとに変換を合成した結果は、リストへの mapcar / remove-if-not の合成と同じ。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (multiple-value-bind (specs records) (transform-random-case seed)
    (equal (list-oracle specs records)
           (transformed-records (mapcar #'spec-transform specs) records))))

(def-source-property transforms/report-whether-the-record-is-kept
  "残ったレコードの 2 つ目の値は t、捨てたレコードは (values nil nil)。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (multiple-value-bind (specs records) (transform-random-case seed)
    (let ((transforms (mapcar #'spec-transform specs)))
      (every (lambda (record)
               (let ((results (multiple-value-list (apply-transforms transforms record))))
                 (if (list-oracle specs (list record))
                     (equal results (list (first (list-oracle specs (list record))) t))
                     (equal results (list nil nil)))))
             records))))

(test (transforms/map-sees-the-whole-record-tree :suite :nabla.small)
  "map はレコードの木をまるごと受けて返す（葉ごとではない）。"
  (is (equal '((2 1) t)
             (multiple-value-list
              (apply-transforms (list (map-transform #'reverse)) (list 1 2))))))

(defparameter *non-functions* (list 3 "evenp")
  "関数でも関数名でもない値。定数を直接渡すとコンパイル時に型の WARNING になるので、変数から渡す。")

(test (transforms/reject-objects-that-are-not-transforms :suite :nabla.small)
  "変換のリストに変換でないものがあると type-error（batch は変換のリストに入れられない）。"
  (signals type-error (apply-transforms (list #'identity) 1))
  (signals type-error (apply-transforms (list (map-transform #'1+) :batch) 1))
  (dolist (not-a-function *non-functions*)
    (signals type-error (map-transform not-a-function))
    (signals type-error (filter-transform not-a-function))))
