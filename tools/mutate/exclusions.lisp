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
  :reason "x = lo のとき、(< x lo) は偽で cond は x をそのまま返し、(<= x lo) は真で lo を返す。x = lo なので lo と x は同じ値であり、どちらの分岐でも返る値は変わらない等価変異体（clamp の下限チェックにだけ現れる、runner のサンプルとしての既知の等価変異体）")
 ;; issue #38（u4, bf16/f16 の RNE 変換）
 (:file "src/float16.lisp"
  :mutation "(DEFUN NABLA::%FLOOR-LOG2 (NABLA::R) \"正の有理数 R について、2^E <= R < 2^(E+1) を満たす整数 E を返す
（floor(log2 R)）。浮動小数点への変換を経由せず、有理数のまま正確に
計算する。\" (LET ((NABLA::E (- (INTEGER-LENGTH (NUMERATOR NABLA::R)) (INTEGER-LENGTH (DENOMINATOR NABLA::R))))) (LOOP NABLA::WHILE (< NABLA::R (EXPT 2 NABLA::E)) DO (DECF NABLA::E)) (LOOP NABLA::WHILE (>= NABLA::R (EXPT 2 (1+ NABLA::E))) DO (INCF NABLA::E)) NABLA::E)) -> (DEFUN NABLA::%FLOOR-LOG2 (NABLA::R) \"正の有理数 R について、2^E <= R < 2^(E+1) を満たす整数 E を返す
（floor(log2 R)）。浮動小数点への変換を経由せず、有理数のまま正確に
計算する。\" (LET ((NABLA::E (+ (INTEGER-LENGTH (NUMERATOR NABLA::R)) (INTEGER-LENGTH (DENOMINATOR NABLA::R))))) (LOOP NABLA::WHILE (< NABLA::R (EXPT 2 NABLA::E)) DO (DECF NABLA::E)) (LOOP NABLA::WHILE (>= NABLA::R (EXPT 2 (1+ NABLA::E))) DO (INCF NABLA::E)) NABLA::E))"
  :reason "この式は E の最終的な正しい値ではなく、その後に続く2つの LOOP（2^E <= R になるまで DECF、2^(E+1) > R になるまで INCF）が正しい floor(log2 R) に収束するための初期値（見積もり）にすぎない。初期値をどんな整数にしても、2つの LOOP が真の値まで補正するので、返り値は変わらない等価変異体（ただし見積もりが大きく外れるほど LOOP の反復回数が増える。パフォーマンス上の意図はあるが正しさには影響しない）"))
