;;;; StableHLO をコンパイルして得た vmfb を、session にロード済みの
;;;; 関数として呼び出す、nabla.iree の公開 API（issue #8）。
;;;;
;;;; ここでは compile-stablehlo / session-append-module を呼び直さない。
;;;; 呼び出し側が compile-stablehlo → session-append-module → invoke の
;;;; 順に組み合わせる（W3 契約 §2）。

(in-package #:nabla.iree)

(defun %check-invoke-argument (argument session)
  "ARGUMENT が invoke に渡せる device-array であることを確かめる。released
なら IREE-OBJECT-RELEASED（kind :device-array）、SESSION の device と
異なる device で作られていれば（v1 では専用のコンディションを設けず）
plain ERROR を signal する。"
  (unless (typep argument 'device-array)
    (error "invoke: 引数は device-array でなければならない: ~S" argument))
  (%live-device-array-pointer argument "invoke")
  (unless (eq (device-array-device argument) (session-device session))
    (error "invoke: 引数の device-array は SESSION の device とは別の device で作られている")))

(defun invoke (session full-name &rest arguments)
  "SESSION にロード済みの FULL-NAME（\"module.main\" のような完全修飾名）の
関数を、ARGUMENTS（device-array のリスト）を入力にして呼び出す。

手順: (1) 各 ARGUMENT が live な device-array で、SESSION の device と
同じ device で作られていることを確かめる。(2) with-call で作った call に
call-push-buffer-view で順に push する。(3) call-invoke する（IREE 側の
shape / dtype / arity の不一致は IREE-STATUS-ERROR（code :invalid-argument）
としてそのまま伝わる。runtime.h の hal.buffer_view.assert と vm の
invocation チェックが検出する）。(4) 出力の個数を
iree_runtime_call_outputs + iree_vm_list_size で数え、call-pop-buffer-view
で1つずつ取り出して %wrap-buffer-view で device-array に包む。

返り値は複数の device-array を多値で返す（出力が無い関数は (values)）。
出力の要素型が *element-types* に無い未知の値なら、すでに取り出した
buffer view を解放してからエラーを signal する。途中で失敗したときは、
それまでに包んだ出力の device-array をすべて release-device-array してから
再度エラーを送出する。

CALL-INVOKE 本体は WITH-ALL-FLOAT-TRAPS-MASKED（float-traps.lisp、issue #53
で SBCL（x86-64）が制御できる5種類全部に広げた）で包む（MAKE-DEVICE の docstring
参照。カーネルを実行するワーカースレッドは MAKE-DEVICE の時点でマスク
済みになるが、呼び出し元スレッド自身がここで結果を読み出す・計算に参加する
場合に備えて同じマスクを及ぼす）。"
  (dolist (argument arguments)
    (%check-invoke-argument argument session))
  (with-all-float-traps-masked
    (with-call (call session full-name)
      (dolist (argument arguments)
        (call-push-buffer-view call (%live-device-array-pointer argument "invoke")))
      (call-invoke call)
      (let* ((outputs (%runtime-call-outputs call))
             (count (%vm-list-size outputs))
             (results nil))
        (handler-case
            (progn
              (dotimes (i count)
                (let* ((buffer-view (call-pop-buffer-view call))
                       (element-type (buffer-view-element-type buffer-view)))
                  (unless (keywordp element-type)
                    (buffer-view-release buffer-view)
                    (error "invoke: ~A の出力の要素型 ~S が未知（*element-types* に無い）"
                           full-name element-type))
                  (let ((aval (nabla:make-aval (buffer-view-shape buffer-view) element-type)))
                    (push (%wrap-buffer-view buffer-view (session-device session) aval) results))))
              (apply #'values (nreverse results)))
          (error (condition)
            (dolist (result results)
              (release-device-array result))
            (error condition)))))))
