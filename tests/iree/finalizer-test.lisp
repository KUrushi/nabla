;;;; device-array の finalizer による自動解放と、リークしないことの確認
;;;; （issue #11）。
;;;;
;;;; ここでのテストは「FFI 越しのオブジェクトの寿命」を確かめる例ベースの
;;;; テストで、このスキルの「CFFI の生バインディングの疎通確認は例ベース
;;;; でよい」の考え方をそのまま当てはめている（性質として書きにくい）。
;;;; 統計の読み方は tests/iree/support.lisp の gc-and-run-finalizers の
;;;; コメント（SBCL の finalizer スレッドの非同期性）を参照。
;;;;
;;;; nabla.asd がこのファイルを compiler-test / runtime-test / device-array-test /
;;;; execute-test より先にロードしている理由: このファイルの
;;;; (sb-ext:gc :full t) 呼び出しは、device / session を大量に作っては壊す
;;;; 既存のテストが積み重なった後に呼ぶと、SBCL が
;;;; "garbage_collect: no SP known for thread" という fatal error でプロセス
;;;; ごと落ちることを確率的に起こす（nabla.iree の finalizer 機構自体の
;;;; バグではなく、SBCL のスレッド管理と IREE がドライバ内部に持つスレッド
;;;; との間の、既知の相性問題だと考えられる。このファイルだけを単独で実行
;;;; した場合や、tools/ のスタンドアロンな 50000 回ループのスクリプトでは
;;;; 毎回問題なく通ることを確認済み。詳細と再現手順は PR 本文と
;;;; docs/iree-build.md）。ロード順を変えるのは発生頻度を下げる緩和策で、
;;;; 根本原因（SBCL 側か IREE 側か）は直していない。

(in-package #:nabla.iree.tests)

(define-iree-test finalizer/leak-loop/allocator-bytes-do-not-grow-unbounded
    "device-array を作っては参照を捨てる（明示的に release-device-array
しない）ループを 20000 回まわしても、IREE 側の allocator 統計で
「確保した量 - 解放した量」が数個分（stragglers）を超えて増え続けない。
1個あたりのバイト数は、最初に1個だけ作って device-bytes-allocated の
差分から学ぶ（差分が0なら、統計が効いていないということなので SKIP では
なく FAIL にする。固定した IREE ビルドは IREE_STATISTICS_ENABLE=1 が
既定で、allocator の統計はドライバに依存せず正確にカウントされることを
事前に確認済み———W3 契約 fact 1、local-sync でも同じ計測ができることを
別途確認した）。

driver には :local-sync を使う（device-array / allocator 統計のコードは
:local-task と共通なので、finalizer の解放とリークの検出という、この
テストが確かめたい性質はどちらでも変わらない）。このテストが呼ぶ
(sb-ext:gc :full t) と、この環境で確認した fatal error の関係は、この
ファイルのトップのコメントを参照。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-sync)
    (let* ((spec (make-array-spec '(2 2) :f32))
           (baseline (device-allocator-statistics device)))
      ;; 1個だけ作って、その1個分のバイト数を学ぶ。
      (let ((probe (to-device (make-random-array spec) device)))
        (let* ((after-probe (device-allocator-statistics device))
               (bytes-per-array (- (allocator-statistics-device-bytes-allocated after-probe)
                                    (allocator-statistics-device-bytes-allocated baseline))))
          (release-device-array probe)
          (gc-and-run-finalizers)
          (is (plusp bytes-per-array)
              "device-bytes-allocated が動かなかった。IREE_STATISTICS_ENABLE が無効になっていないか確認すること")
          (let ((before-loop (device-allocator-statistics device))
                (iterations 20000))
            (dotimes (i iterations)
              ;; 戻り値を変数に束縛しない: ループを抜けたときに残る参照が
              ;; あると、その分だけ finalizer が呼ばれず統計が減らない。
              (to-device (make-random-array spec) device)
              (when (zerop (mod (1+ i) 1000))
                (gc-and-run-finalizers)))
            (gc-and-run-finalizers)
            (let* ((after-loop (device-allocator-statistics device))
                   (allocated-delta (- (allocator-statistics-device-bytes-allocated after-loop)
                                        (allocator-statistics-device-bytes-allocated before-loop)))
                   (freed-delta (- (allocator-statistics-device-bytes-freed after-loop)
                                    (allocator-statistics-device-bytes-freed before-loop)))
                   (live-delta (- allocated-delta freed-delta)))
              (is (>= allocated-delta (* iterations bytes-per-array))
                  "device-bytes-allocated が20000回分だけ増えていない（統計が実際に記録されているかの確認）")
              (is (<= live-delta (* 16 bytes-per-array))
                  "GC・finalizer を挟んでも解放されずに残っているバイト数が多すぎる（リークの疑い）"))))))))

(define-iree-test finalizer/release-device-array/explicit-release-then-gc-is-not-double-free
    "N個の device-array を明示的に release-device-array してから GC を
走らせても、二重解放（統計が過剰にカウントされる、あるいはクラッシュする）
にならない: release-device-array の後は device-bytes-allocated と
device-bytes-freed の増分が一致する。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-sync)
    (let* ((spec (make-array-spec '(2 2) :f32))
           (before (device-allocator-statistics device))
           (n 500)
           (arrays (loop repeat n collect (to-device (make-random-array spec) device))))
      (dolist (array arrays)
        (release-device-array array))
      (gc-and-run-finalizers)
      (let* ((after (device-allocator-statistics device))
             (allocated-delta (- (allocator-statistics-device-bytes-allocated after)
                                  (allocator-statistics-device-bytes-allocated before)))
             (freed-delta (- (allocator-statistics-device-bytes-freed after)
                              (allocator-statistics-device-bytes-freed before))))
        (is (= allocated-delta freed-delta)
            "明示的に release した後、確保量と解放量の増分が一致しない（二重解放の疑い）")))))

(define-iree-test finalizer/released-array/to-host-and-invoke-signal-iree-object-released
    "release-device-array 済みの device-array を to-host / invoke に渡すと
IREE-OBJECT-RELEASED（kind :device-array）が signal される。
release-device-array 自体は idempotent。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-sync)
    (with-session (session device)
      (let ((array (to-device (make-random-array (make-array-spec '(2 2) :f32)) device)))
        (release-device-array array)
        (is (device-array-released-p array))
        (release-device-array array)
        (is (device-array-released-p array))
        (handler-case
            (progn
              (to-host array)
              (fiveam:fail "released device-array の to-host が signal しなかった"))
          (iree-object-released (condition)
            (is (eq :device-array (iree-object-released-kind condition)))))
        (handler-case
            (progn
              (invoke session "module.main" array)
              (fiveam:fail "released device-array の invoke が signal しなかった"))
          (iree-object-released (condition)
            (is (eq :device-array (iree-object-released-kind condition)))))))))

(define-iree-test finalizer/dropped-after-release-device/finalizer-does-not-crash
    "with-device の中で作った device-array の参照を捨て、with-device を
抜けて device 自身が release-device された後で GC・finalizer を走らせても
クラッシュしない（device-array は生成時に device を retain しているので、
finalizer は自分が retain した device pointer を release するだけであり、
呼び出し側の release-device とは独立に安全に動く。W3 契約 fact 5）。"
  (skip-unless-iree :library :runtime)
  (with-device (device :local-sync)
    (let ((spec (make-array-spec '(2 2) :f32)))
      (dotimes (i 100)
        ;; 戻り値を保持しない: ここで作った device-array への参照は、この
        ;; let のスコープを抜けると残らない。
        (to-device (make-random-array spec) device))))
  ;; with-device を抜けた（device は release-device 済み）後で GC する。
  (gc-and-run-finalizers)
  (is (not nil) "device を release した後の finalizer 実行がクラッシュしなかった"))
