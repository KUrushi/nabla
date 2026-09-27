;;;; vmfb ディスクキャッシュ（issue #10）を find-backend :iree 経由で
;;;; 確かめるテスト。IREE の共有ライブラリが要るので、compile-cache-test.lisp
;;;; （nabla/tests、フェイクだけで確かめる版）とは別にここへ置く。
;;;;
;;;; NB:*COMPILE-CACHE-DIRECTORY* は必ず一時ディレクトリに束縛し、
;;;; ~/.cache には触らない。

(in-package #:nabla.iree.tests)

(define-iree-test compile-cache/iree/second-compile-returns-identical-bytes
    "同じ StableHLO テキストを一時キャッシュディレクトリで2回
backend-compile すると、1回目・2回目のバイト列は equalp で一致する
（2回目はディスクキャッシュから読んでいる。IREE の実際のコンパイルが
何度呼ばれたかは iree-backend からは数えられないので、ここではバイト
列の一致だけを確かめる）。"
  (skip-unless-iree :library :both)
  (with-temporary-directory (dir)
    (let* ((nabla:*compile-cache-directory* dir)
           (backend (nabla:find-backend :iree))
           (text (stablehlo-fixture "add"))
           (first (nabla:backend-compile backend text))
           (second (nabla:backend-compile backend text)))
      (is (equalp first second)))))

(define-iree-test compile-cache/iree/cached-bytes-run-and-match-reference
    "一時キャッシュディレクトリで2回目に読んだ（ディスクキャッシュ経由の）
バイト列を backend-load → backend-invoke した結果は、reference-add の
期待値と allclose :dtype :f32 で一致する（キャッシュから読んだ vmfb が
実際に実行できることの確認）。"
  (skip-unless-iree :library :both)
  (with-temporary-directory (dir)
    (let* ((nabla:*compile-cache-directory* dir)
           (backend (nabla:find-backend :iree))
           (text (stablehlo-fixture "add")))
      (nabla:backend-compile backend text)
      (let* ((cached (nabla:backend-compile backend text))
             (module (nabla:backend-load backend cached))
             (a (make-random-array (make-array-spec '(4 8) :f32) :seed 0))
             (b (make-random-array (make-array-spec '(4 8) :f32) :seed 1)))
        (unwind-protect
             (with-device-arrays ((da (to-device a backend))
                                  (db (to-device b backend)))
               (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                 (is (allclose (to-host result) (reference-add a b) :dtype :f32))))
          (nabla:backend-unload backend module))))))
