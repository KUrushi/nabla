;;;; skip-unless-cuda / skip-unless-iree のマクロ展開そのものを確かめる
;;;; テスト。
;;;;
;;;; skip-unless-cuda は fiveam:fail / fiveam:skip のどちらの枝でも
;;;; (return-from iree-test ...) して DEFINE-IREE-TEST / DEFINE-IREE-TEST/LARGE
;;;; の (block iree-test ...) から抜けなければならない。fiveam:fail は
;;;; process-failure を呼ぶだけで非局所脱出しないため、return-from を
;;;; 省いた枝があると、その枝を通った後もテスト本体がそのまま実行され
;;;;続けてしまう（実際に、NABLA_REQUIRE_CUDA=1 かつ CUDA が無い環境で
;;;; cross-device テストが cuda backend の作成や check-it の実行まで
;;;; 進み、check-it が失敗例を tests/regressions/ に書き出す、という形で
;;;; 起きた）。

(in-package #:nabla.iree.tests)

(defun %count-return-from-iree-test (form)
  "FORM の中に現れる (return-from iree-test ...) の個数を、コンス木を
再帰的に辿って数える（マクロ展開結果の構造をそのまま検査するための
ヘルパー。文字列の見た目ではなく S式の構造を見るので、フォーマットの
変更に左右されない）。"
  (cond
    ((and (consp form) (eq (first form) 'return-from) (eq (second form) 'iree-test))
     1)
    ((consp form)
     (+ (%count-return-from-iree-test (car form))
        (%count-return-from-iree-test (cdr form))))
    (t 0)))

(fiveam:test (support/skip-unless-cuda/both-branches-return-from-iree-test
              :suite :nabla.medium)
    "SKIP-UNLESS-CUDA の展開は、cuda が使えないときの2つの枝（
NABLA_REQUIRE_CUDA が立っているときの fiveam:fail と、立っていないときの
fiveam:skip）のどちらでも (return-from iree-test) する。どちらか一方でも
欠けると、DEFINE-IREE-TEST / DEFINE-IREE-TEST/LARGE の block から抜けずに
テスト本体の残りが実行されてしまう回帰を防ぐ。"
  (is (= 2 (%count-return-from-iree-test (macroexpand-1 '(skip-unless-cuda))))))

(fiveam:test (support/skip-unless-iree/both-branches-return-from-iree-test
              :suite :nabla.medium)
    "SKIP-UNLESS-IREE も同じ形（fiveam:fail の枝・fiveam:skip の枝の両方が
return-from iree-test する）であることを、SKIP-UNLESS-CUDA と同じ検査で
確かめる（こちらは元から正しいが、同じ構造なので同じ性質を守る）。"
  (is (= 2 (%count-return-from-iree-test (macroexpand-1 '(skip-unless-iree))))))
