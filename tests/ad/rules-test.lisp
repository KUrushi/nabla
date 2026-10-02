;;;; jvp / transpose ルールの登録・コンディション・def-jvp-partials（issue #77、77a）。
;;;;
;;;; ここで使うプリミティブは %TEST-AD- 接頭辞のテスト専用。77c が本物の
;;;; プリミティブや %test-add / %test-neg に足すルールとは干渉させない。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %test-ad-same-aval (in-avals)
  (first in-avals))

;; ルール無し。
(nb:defprimitive %test-ad-bare ()
  :abstract-eval #'%test-ad-same-aval)

;; パラメタ付き。
(nb:defprimitive %test-ad-scale (:factor)
  :abstract-eval (lambda (in-avals &key factor) (declare (ignore factor)) (first in-avals)))

;; 二項（def-jvp-partials 用）。値は a * b として扱う。
(nb:defprimitive %test-ad-prod ()
  :abstract-eval #'%test-ad-same-aval)

(defun %reset-rules (name)
  "NAME のプリミティブの jvp / transpose スロットを NIL に戻す（各テストが
自分でフィクスチャを整えるため。順序や再実行に依存させない）。"
  (let ((prim (nb::find-primitive name)))
    (setf (nb::primitive-jvp prim) nil
          (nb::primitive-transpose prim) nil)
    prim))

(test ad-rules/def-jvp-rule-sets-slot
  "def-jvp-rule は primitive-jvp に関数を入れ、その関数は
(primals out tangents &key params) の規約で呼べる。"
  (let ((prim (%reset-rules :%test-ad-scale)))
    (is (null (nb::primitive-jvp prim)))
    (nb::def-jvp-rule %test-ad-scale (primals out tangents &key factor)
      (list :jvp primals out tangents factor))
    (let ((rule (nb::primitive-jvp prim)))
      (is (functionp rule))
      (is (equal '(:jvp (p) o (tg) 3)
                 (apply rule '(p) 'o '(tg) '(:factor 3)))))))

(test ad-rules/def-transpose-rule-sets-slot
  "def-transpose-rule は primitive-transpose に関数を入れ、その関数は
(ct invars &key params) の規約で呼べる。jvp スロットには触らない。"
  (let ((prim (%reset-rules :%test-ad-scale))
        (jvp (lambda (primals out tangents) (declare (ignore primals out tangents)) nil)))
    (setf (nb::primitive-jvp prim) jvp)
    (nb::def-transpose-rule %test-ad-scale (ct invars &key factor)
      (list :transpose ct invars factor))
    (is (equal '(:transpose c (i) 5)
               (apply (nb::primitive-transpose prim) 'c '(i) '(:factor 5))))
    (is (eq jvp (nb::primitive-jvp prim)))))

(test ad-rules/unregistered-name-is-an-error
  "未登録のプリミティブ名への def-jvp-rule / def-transpose-rule は、ルールを
定義する時点で unknown-primitive になる。"
  (signals nb:unknown-primitive
    (eval '(nb::def-jvp-rule %test-ad-does-not-exist (primals out tangents) nil)))
  (signals nb:unknown-primitive
    (eval '(nb::def-transpose-rule %test-ad-does-not-exist (ct invars) nil)))
  (is (null (nb::find-primitive :%test-ad-does-not-exist))))

(test ad-rules/defprimitive-accepts-rule-keys
  "defprimitive の :jvp / :transpose は対応するスロットに入る。省略すると NIL。"
  (let ((jvp (lambda (primals out tangents) (declare (ignore primals out tangents)) :j))
        (transpose (lambda (ct invars) (declare (ignore ct invars)) :t)))
    (nb:defprimitive %test-ad-with-rules ()
      :abstract-eval #'%test-ad-same-aval
      :jvp jvp
      :transpose transpose)
    (let ((prim (nb::find-primitive :%test-ad-with-rules)))
      (is (eq jvp (nb::primitive-jvp prim)))
      (is (eq transpose (nb::primitive-transpose prim)))))
  (let ((prim (nb::find-primitive :%test-ad-bare)))
    (is (null (nb::primitive-jvp prim)))
    (is (null (nb::primitive-transpose prim)))))

(test ad-rules/require-rule-signals-no-rule-conditions
  "ルールが無いプリミティブは no-jvp-rule / no-transpose-rule（どちらも
autodiff-error）を、name つきで signal する。ルールがあればその関数を返す。"
  (let ((prim (nb::find-primitive :%test-ad-bare)))
    (handler-case (progn (nb::require-jvp-rule prim) (fail "no-jvp-rule が signal されなかった"))
      (nb:no-jvp-rule (c)
        (is (typep c 'nb:autodiff-error))
        (is (eq :%test-ad-bare (nb:no-jvp-rule-name c)))
        (is (search "%TEST-AD-BARE" (princ-to-string c)))))
    (handler-case (progn (nb::require-transpose-rule prim) (fail "no-transpose-rule が signal されなかった"))
      (nb:no-transpose-rule (c)
        (is (typep c 'nb:autodiff-error))
        (is (eq :%test-ad-bare (nb:no-transpose-rule-name c))))))
  (let ((prim (nb::find-primitive :%test-ad-with-rules)))
    (is (eq (nb::primitive-jvp prim) (nb::require-jvp-rule prim)))
    (is (eq (nb::primitive-transpose prim) (nb::require-transpose-rule prim)))))

(test ad-rules/defprimitive-redefinition-keeps-rules
  "defprimitive を再評価しても、後から付けたルールは引き継がれる。:jvp /
:transpose を明示すればそちらで上書きする。"
  (nb:defprimitive %test-ad-redefine ()
    :abstract-eval #'%test-ad-same-aval)
  (%reset-rules :%test-ad-redefine)
  (nb::def-jvp-rule %test-ad-redefine (primals out tangents) (declare (ignore primals out tangents)) :old-jvp)
  (nb::def-transpose-rule %test-ad-redefine (ct invars) (declare (ignore ct invars)) :old-transpose)
  (let ((old-jvp (nb::primitive-jvp (nb::find-primitive :%test-ad-redefine))))
    (nb:defprimitive %test-ad-redefine ()
      :abstract-eval #'%test-ad-same-aval)
    (let ((prim (nb::find-primitive :%test-ad-redefine)))
      (is (eq old-jvp (nb::primitive-jvp prim)))
      (is (functionp (nb::primitive-transpose prim))))
    (let ((new (lambda (primals out tangents) (declare (ignore primals out tangents)) :new)))
      (nb:defprimitive %test-ad-redefine ()
        :abstract-eval #'%test-ad-same-aval
        :jvp new)
      (let ((prim (nb::find-primitive :%test-ad-redefine)))
        (is (eq new (nb::primitive-jvp prim)))
        (is (functionp (nb::primitive-transpose prim)))))))

;;; --- def-jvp-partials ---

(defun %test-ad-prod-rule ()
  "(a, b) ↦ a * b の偏微分 b, a から作った jvp ルール。"
  (nb::make-jvp-from-partials
   (list (lambda (primals out) (declare (ignore out)) (second primals))
         (lambda (primals out) (declare (ignore out)) (first primals)))))

(defun %trace-prod-jvp (aval ta-zero-p tb-zero-p)
  "(a b ta tb) を引数に、%test-ad-prod の partials ルールを呼ぶ関数をトレースした
graph を返す。TA-ZERO-P / TB-ZERO-P が真ならその接線は symbolic zero（graph の
入力には ta / tb のうち非ゼロのものだけが入る）。"
  (let ((rule (%test-ad-prod-rule))
        (zero (nb::make-symbolic-zero aval)))
    (cond
      ((and ta-zero-p tb-zero-p)
       (nb:trace-to-graph (nb:with-tracing (a b) (funcall rule (list a b) a (list zero zero)))
                          (list aval aval)))
      (ta-zero-p
       (nb:trace-to-graph (nb:with-tracing (a b tb) (funcall rule (list a b) a (list zero tb)))
                          (list aval aval aval)))
      (tb-zero-p
       (nb:trace-to-graph (nb:with-tracing (a b ta) (funcall rule (list a b) a (list ta zero)))
                          (list aval aval aval)))
      (t
       (nb:trace-to-graph (nb:with-tracing (a b ta tb) (funcall rule (list a b) a (list ta tb)))
                          (list aval aval aval aval))))))

(test ad-rules/partials-sums-only-nonzero-tangent-terms
  "def-jvp-partials 相当のルールは、非ゼロ接線の項 (偏微分 * 接線) だけを足す。
両方非ゼロなら b*ta + a*tb。片方がゼロなら mul 1つだけ（add は無い）。
どちらもゼロなら出力と同じ aval の symbolic zero。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :f64) :max-rank 3 :max-dim 4)
                                  (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((aval (%spec-aval spec))
                           (dtype (array-spec-dtype spec))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (+ seed 1)))
                           (ta (make-random-array spec :seed (+ seed 2)))
                           (tb (make-random-array spec :seed (+ seed 3)))
                           (both (%trace-prod-jvp aval nil nil))
                           (only-ta (%trace-prod-jvp aval nil t))
                           (only-tb (%trace-prod-jvp aval t nil)))
                      (flet ((prims (graph)
                               (mapcar (lambda (e) (nb:primitive-name (nb:eqn-prim e)))
                                       (nb:graph-eqns graph))))
                        (and
                         (allclose (nb:eval-graph both a b ta tb)
                                   (nb::%t-add (nb::%t-mul b ta) (nb::%t-mul a tb))
                                   :dtype dtype)
                         (equal '(:mul :mul :add) (prims both))
                         (allclose (nb:eval-graph only-ta a b ta) (nb::%t-mul b ta) :dtype dtype)
                         (equal '(:mul) (prims only-ta))
                         (allclose (nb:eval-graph only-tb a b tb) (nb::%t-mul a tb) :dtype dtype)
                         (equal '(:mul) (prims only-tb)))))))
                :regression-id ad-rules/partials-sums-only-nonzero-tangent-terms
                :regression-file (regression-path "ad-rules-partials-nonzero"))))

(test ad-rules/partials-all-zero-returns-symbolic-zero-of-out-aval
  "全接線がゼロなら、eqn を足さず、出力の aval の symbolic zero を返す。"
  (let* ((aval (nb:make-aval '(2 3) :f32))
         (result nil)
         (rule (%test-ad-prod-rule))
         (zero (nb::make-symbolic-zero aval))
         (recorder (lambda (r) (setf result r) nil)))
    (let ((graph (nb:trace-to-graph
                  (nb:with-tracing (a b)
                    (funcall recorder (funcall rule (list a b) a (list zero zero)))
                    a)
                  (list aval aval))))
      (is (null (nb:graph-eqns graph))))
    (is (nb::symbolic-zero-p result))
    (is (equalp aval (nb::symbolic-zero-aval result)))))

(test ad-rules/partials-receive-params
  "偏微分関数は (primals out &key params) で呼ばれ、eqn のパラメタが渡る。"
  (let* ((aval (nb:make-aval '(2) :f32))
         (seen nil)
         (rule (nb::make-jvp-from-partials
                (list (lambda (primals out &key factor)
                        (declare (ignore primals))
                        (setf seen (list :factor factor))
                        out))))
         (zero-free (lambda (a ta) (funcall rule (list a) a (list ta) :factor 7))))
    (nb:trace-to-graph (nb:with-tracing (a ta) (funcall zero-free a ta)) (list aval aval))
    (is (equal '(:factor 7) seen))))

(test ad-rules/def-jvp-partials-installs-rule
  "def-jvp-partials は偏微分関数のリストから作ったルールを primitive-jvp に入れる。"
  (nb::def-jvp-partials %test-ad-prod
    (lambda (primals out) (declare (ignore out)) (second primals))
    (lambda (primals out) (declare (ignore out)) (first primals)))
  (let* ((aval (nb:make-aval '(3) :f64))
         (rule (nb::primitive-jvp (nb::find-primitive :%test-ad-prod)))
         (graph (nb:trace-to-graph
                 (nb:with-tracing (a b ta tb) (funcall rule (list a b) a (list ta tb)))
                 (list aval aval aval aval)))
         (a (make-array 3 :element-type 'double-float :initial-contents '(1d0 2d0 3d0)))
         (b (make-array 3 :element-type 'double-float :initial-contents '(4d0 5d0 6d0)))
         (ta (make-array 3 :element-type 'double-float :initial-contents '(1d0 1d0 1d0)))
         (tb (make-array 3 :element-type 'double-float :initial-contents '(2d0 0d0 -1d0))))
    ;; b*ta + a*tb
    (is (equalp #(6d0 5d0 3d0) (nb:eval-graph graph a b ta tb)))))

(test ad-rules/partials-rank0-coefficient-broadcasts
  "偏微分が rank 0 のトレーサ（や実数）でも、%t-mul が接線の shape へ
ブロードキャストするので、出力の接線は out と同じ aval を持ち、値は 係数 * 接線。"
  (let* ((aval (nb:make-aval '(2 3) :f32))
         (rule (nb::make-jvp-from-partials
                (list (lambda (primals out)
                        (declare (ignore primals out))
                        (nb::%lift-number-to 2.0 :f32 '())))))
         (graph (nb:trace-to-graph
                 (nb:with-tracing (a ta) (funcall rule (list a) a (list ta)))
                 (list aval aval)))
         (ta (make-array '(2 3) :element-type 'single-float :initial-element 1.5)))
    (is (equalp aval (nb:var-aval (first (nb:graph-outvars graph)))))
    (is (allclose (nb:eval-graph graph ta ta)
                  (make-array '(2 3) :element-type 'single-float :initial-element 3.0)
                  :dtype :f32))))

(test ad-rules/partials-are-not-called-for-zero-tangents-and-zero-partials-skip
  "ゼロ接線の入力の偏微分関数は呼ばれない。偏微分が symbolic-zero を返したら
その項は飛ばす。"
  (let* ((aval (nb:make-aval '(2) :f32))
         (zero (nb::make-symbolic-zero aval))
         (calls (list 0 0))
         (rule (nb::make-jvp-from-partials
                (list (lambda (primals out) (declare (ignore out))
                        (incf (first calls)) (first primals))
                      (lambda (primals out) (declare (ignore primals))
                        (incf (second calls)) (nb::make-symbolic-zero (nb::tracer-aval out))))))
         (result nil)
         (recorder (lambda (r) (setf result r) nil)))
    (nb:trace-to-graph
     (nb:with-tracing (a ta tb)
       (funcall recorder (funcall rule (list a a) a (list ta zero)))
       (funcall recorder (funcall rule (list a a) a (list zero tb)))
       a)
     (list aval aval aval))
    (is (equal '(1 1) calls))
    (is (nb::symbolic-zero-p result))))
