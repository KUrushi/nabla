;;;; nb:aval 回りの性質（issue #7）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(test aval/array-aval/round-trips-spec-shape-and-dtype
  "array-aval は make-random-array で作った配列から、元の spec と同じ
shape・dtype を持つ aval を作る。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let* ((array (make-random-array spec))
                         (aval (nb:array-aval array (array-spec-dtype spec))))
                    (and (equal (nb:aval-shape aval) (array-spec-shape spec))
                         (eq (nb:aval-dtype aval) (array-spec-dtype spec)))))
                :regression-id aval/array-aval/round-trips-spec-shape-and-dtype
                :regression-file (regression-path "aval-array-aval-round-trips"))))

(test aval/aval-size/matches-array-total-size
  "aval-size は元の配列の array-total-size と一致する（rank 0 でも1）。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let* ((array (make-random-array spec))
                         (aval (nb:array-aval array (array-spec-dtype spec))))
                    (= (nb:aval-size aval) (array-total-size array))))
                :regression-id aval/aval-size/matches-array-total-size
                :regression-file (regression-path "aval-size-matches-array-total-size"))))

(test aval/aval-byte-length/is-size-times-dtype-byte-width
  "aval-byte-length は aval-size と dtype-byte-width の積。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let* ((array (make-random-array spec))
                         (aval (nb:array-aval array (array-spec-dtype spec))))
                    (= (nb:aval-byte-length aval)
                       (* (nb:aval-size aval)
                          (nb:dtype-byte-width (array-spec-dtype spec))))))
                :regression-id aval/aval-byte-length/is-size-times-dtype-byte-width
                :regression-file (regression-path "aval-byte-length-is-size-times-width"))))

(test aval/aval-rank/matches-shape-length
  "aval-rank は shape の長さに一致する。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let* ((array (make-random-array spec))
                         (aval (nb:array-aval array (array-spec-dtype spec))))
                    (= (nb:aval-rank aval) (length (array-spec-shape spec)))))
                :regression-id aval/aval-rank/matches-shape-length
                :regression-file (regression-path "aval-rank-matches-shape-length"))))

(test aval/aval-byte-length/i1-equals-size
  "make-aval '(2 3) :i1 の aval-byte-length は aval-size と一致する
（:i1 のバイト幅は1。issue #37）。"
  (let ((aval (nb:make-aval '(2 3) :i1)))
    (is (= (nb:aval-byte-length aval) (nb:aval-size aval)))))

(test aval/make-aval/rejects-non-list-shape
  "make-aval は shape がリストでなければエラーを signal する。"
  (signals error (nb:make-aval 3 :f32)))

(test aval/make-aval/rejects-negative-dimension
  "make-aval は負の次元を含む shape を拒否する。"
  (signals error (nb:make-aval '(2 -1) :f32)))

(test aval/make-aval/rejects-unknown-dtype
  "make-aval は DTYPE 型でないキーワードを拒否する。"
  (signals error (nb:make-aval '(2 3) :not-a-dtype)))

(test aval/aval-p/only-avals-satisfy
  "aval-p は make-aval で作った値だけ真を返す。"
  (is (nb:aval-p (nb:make-aval '(2 3) :f32)))
  (is (not (nb:aval-p (list '(2 3) :f32)))))

(test aval/equalp/compares-by-value
  "同じ shape・dtype の aval は equalp で等しい。専用の AVAL= は export
しない（ハイラムの法則、公開シンボルは最小限にする）。"
  (is (equalp (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)))
  (is (not (equalp (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f64))))
  (is (not (equalp (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 2) :f32)))))
