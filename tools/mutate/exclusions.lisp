;;;; exclusions.lisp -- mutation testing の除外リスト（データファイル）
;;;;
;;;; 生き残った変異体が等価変異体だと判断できたときに、理由をつけて
;;;; ここに追記する。書き方は .claude/skills/nabla-testing/references/mutation.md
;;;; の「4. 生き残った変異体への対処」と tools/mutate/README.md を見る。
;;;;
;;;; トップレベルのフォームは1つだけ: 次の形の plist のリスト。
;;;;
;;;;   (:file "src/core/primitives/add.lisp"
;;;;    :form "(defprimitive add ...)"
;;;;    :mutation "(* 1 x) -> (/ x 1)"
;;;;    :reason "x に 1 をかけても割っても値が変わらない等価変異体")

((:file "tools/mutate/sample/src/sample.lisp"
  :mutation "(DEFUN NABLA.MUTATE.SAMPLE:CLAMP (NABLA.MUTATE.SAMPLE::X NABLA.MUTATE.SAMPLE::LO NABLA.MUTATE.SAMPLE::HI) \"X を [LO, HI] に収める。\" (COND ((< NABLA.MUTATE.SAMPLE::X NABLA.MUTATE.SAMPLE::LO) NABLA.MUTATE.SAMPLE::LO) ((> NABLA.MUTATE.SAMPLE::X NABLA.MUTATE.SAMPLE::HI) NABLA.MUTATE.SAMPLE::HI) (T NABLA.MUTATE.SAMPLE::X))) -> (DEFUN NABLA.MUTATE.SAMPLE:CLAMP (NABLA.MUTATE.SAMPLE::X NABLA.MUTATE.SAMPLE::LO NABLA.MUTATE.SAMPLE::HI) \"X を [LO, HI] に収める。\" (COND ((<= NABLA.MUTATE.SAMPLE::X NABLA.MUTATE.SAMPLE::LO) NABLA.MUTATE.SAMPLE::LO) ((> NABLA.MUTATE.SAMPLE::X NABLA.MUTATE.SAMPLE::HI) NABLA.MUTATE.SAMPLE::HI) (T NABLA.MUTATE.SAMPLE::X)))"
  :reason "x = lo のとき、(< x lo) は偽で cond は x をそのまま返し、(<= x lo) は真で lo を返す。x = lo なので lo と x は同じ値であり、どちらの分岐でも返る値は変わらない等価変異体（clamp の下限チェックにだけ現れる、runner のサンプルとしての既知の等価変異体）"))
