;;;; registry-test: フェーズ1の19個のプリミティブがすべて abstract-eval /
;;;; emit / eager の3点セットを持つことを確かめる（issue #31 完了条件、p6）。
;;;;
;;;; p6 は #31 の最後の PR なので、ここで p1〜p6 の全プリミティブを一度に
;;;; 検査する。main に p3 がまだマージされていないと :compare / :select /
;;;; :convert が見つからず失敗するので、このテストが通ることが「#31 の
;;;; すべての子 PR がマージ済み」の合図になる（ガイダンス参照）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defparameter *phase1-primitive-names*
  '(:add :sub :mul :div :max :min :neg :exp :log :tanh
    :compare :select :convert :reshape :broadcast-in-dim :transpose
    :dot-general :reduce-sum :reduce-max)
  "フェーズ1（issue #31）で defprimitive するプリミティブ名の全19個。")

(defun %registry-cases ()
  "*PHASE1-PRIMITIVE-NAMES* の各名前について、make-eqn に渡せる妥当な
(in-avals params) を1組ずつ持つ (name in-avals params) のリストを返す。"
  (list
   (list :add (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)) '())
   (list :sub (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)) '())
   (list :mul (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)) '())
   (list :div (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)) '())
   (list :max (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)) '())
   (list :min (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)) '())
   (list :neg (list (nb:make-aval '(2 3) :f32)) '())
   (list :exp (list (nb:make-aval '(2 3) :f32)) '())
   (list :log (list (nb:make-aval '(2 3) :f32)) '())
   (list :tanh (list (nb:make-aval '(2 3) :f32)) '())
   (list :compare (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)) '(:direction :lt))
   (list :select (list (nb:make-aval '(2 3) :i1) (nb:make-aval '(2 3) :f32) (nb:make-aval '(2 3) :f32)) '())
   (list :convert (list (nb:make-aval '(2 3) :f32)) '(:dtype :bf16))
   (list :reshape (list (nb:make-aval '(2 3) :f32)) '(:shape (3 2)))
   (list :broadcast-in-dim (list (nb:make-aval '(3) :f32)) '(:shape (2 3) :dims (1)))
   (list :transpose (list (nb:make-aval '(2 3) :f32)) '(:perm (1 0)))
   (list :dot-general (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(3 2) :f32))
         '(:lhs-contracting (1) :rhs-contracting (0) :lhs-batch () :rhs-batch ()))
   (list :reduce-sum (list (nb:make-aval '(4 8) :f32)) '(:axes (1)))
   (list :reduce-max (list (nb:make-aval '(4 8) :f32)) '(:axes (1)))))

(test registry/case-list-covers-exactly-the-phase1-names
  "%REGISTRY-CASES と *PHASE1-PRIMITIVE-NAMES* は同じ19個の名前の集合を
指す（このテストファイル自身が名前を書き漏らしていないことの確認）。"
  (let ((case-names (mapcar #'first (%registry-cases))))
    (is (= 19 (length *phase1-primitive-names*)))
    (is (= 19 (length case-names)))
    (is (null (set-difference *phase1-primitive-names* case-names)))
    (is (null (set-difference case-names *phase1-primitive-names*)))))

(test registry/all-phase1-primitives-have-abstract-eval-emit-and-eager
  "*PHASE1-PRIMITIVE-NAMES* のすべてが登録済みで、emit・eager が両方とも
non-NIL である（abstract-eval は defprimitive が必須にしているので、
find-primitive が非NILならすでに non-NIL）。"
  (dolist (name *phase1-primitive-names*)
    (let ((prim (nb::find-primitive name)))
      (is (not (null prim)) (format nil "~S が未登録: main に p1〜p5 がマージされているか確認する" name))
      (when prim
        (is (not (null (nb::primitive-abstract-eval prim))) (format nil "~S に abstract-eval が無い" name))
        (is (not (null (nb::primitive-emit prim))) (format nil "~S に emit が無い" name))
        (is (not (null (nb::primitive-eager prim))) (format nil "~S に eager が無い" name))))))

(test registry/make-eqn-succeeds-for-every-phase1-primitive
  "*PHASE1-PRIMITIVE-NAMES* の全19個について、妥当な params を渡した
make-eqn が例外を出さずに成功する（params の宣言と呼び出し規約の drift
を検出する。wave 3 のトレーサが使う呼び出し方そのもの——契約 §5）。"
  (dolist (case (%registry-cases))
    (destructuring-bind (name in-avals params) case
      (let ((vars (mapcar #'nb::make-var in-avals)))
        (is (not (null (apply #'nb::make-eqn name vars params)))
            (format nil "~S の make-eqn が失敗した" name))))))

;;; --- 自動微分のルール（issue #86。フェーズ2の全ルール検査） ---

(defparameter *non-differentiable-primitive-names* '(:while-loop :cond)
  "意図的に :jvp を持たない本物のプリミティブ。:jvp の無い defprimitive を足すときは、
ここに理由つきで足すか、jvp ルールを書く。
  - :while-loop（issue #131）: 逆モードは反復回数が分からず対応しない。jvp は #134 で足す
    （足したらここから外す）。それまでは grad を通すと NO-JVP-RULE になる。
  - :cond（issue #130）: jvp は #134 で足す（足したらここから外す）。
  ※ :scan（issue #132）は jvp を #135 で足したので外した。逆モード（linearize / transpose）は #139。")

(defun %registered-real-primitives ()
  "登録済みのプリミティブのうち、テスト専用でない（名前が % で始まらない）ものの
リスト。テスト専用のプリミティブ（%test-* / %iree-vjp-* など）は規約で % 接頭辞を
付けるので、ルールの有無を検査する対象から除く。"
  (let ((result '()))
    (maphash (lambda (name prim)
               (unless (char= #\% (char (symbol-name name) 0)) (push prim result)))
             nb::*primitives*)
    result))

(defparameter *linear-primitive-names*
  '(:add :sub :neg :convert :reshape :transpose :broadcast-in-dim :reduce-sum
    :select :mul :div :dot-general)
  "接線について線形に使われうる（:transpose を持つべき）プリミティブ。mul / div /
dot-general は片側だけが線形、select は条件以外の分岐が線形。max / min / exp / log /
tanh / compare / reduce-max / stop-gradient の jvp は接線について線形な式（mul、select
など）だけを出すので、transpose ルールは要らない。")

(test registry/every-registered-primitive-has-a-jvp-rule-or-is-excluded
  "登録されたすべての（テスト専用でない）プリミティブが :jvp を持つか、
*NON-DIFFERENTIABLE-PRIMITIVE-NAMES* に入っている。フェーズ1の19個と stop-gradient は必ず登録済み。"
  (dolist (name (append *phase1-primitive-names* '(:stop-gradient)))
    (is (not (null (nb::find-primitive name))) (format nil "~S が未登録" name)))
  (dolist (prim (%registered-real-primitives))
    (let ((name (nb::primitive-name prim)))
      (is (or (nb::primitive-jvp prim) (member name *non-differentiable-primitive-names*))
          (format nil "~S に jvp ルールが無い（除外するなら *non-differentiable-primitive-names* に足す）" name)))))

(test registry/every-primitive-with-a-transpose-rule-is-listed-as-linear
  "逆向きの検査: :transpose を持つ（テスト専用でない）プリミティブはすべて *LINEAR-PRIMITIVE-NAMES* に入っている。"
  (dolist (prim (%registered-real-primitives))
    (when (nb::primitive-transpose prim)
      (is (member (nb::primitive-name prim) *linear-primitive-names*)
          (format nil "~S は transpose ルールを持つが *linear-primitive-names* に無い" (nb::primitive-name prim))))))

(test registry/every-linear-primitive-has-a-transpose-rule
  "線形なすべてのプリミティブが :transpose を持つ（逆伝播がルール無しで途中で落ちない）。"
  (dolist (name *linear-primitive-names*)
    (let ((prim (nb::find-primitive name)))
      (is (not (null prim)) (format nil "~S が未登録" name))
      (when prim
        (is (not (null (nb::primitive-transpose prim))) (format nil "~S に transpose ルールが無い" name))))))

(test registry/rule-lists-are-consistent
  "線形なプリミティブは、除外リストに入っていない（微分できる）プリミティブ（リストの書き間違いの検出）。"
  (is (null (intersection *linear-primitive-names* *non-differentiable-primitive-names*))))
