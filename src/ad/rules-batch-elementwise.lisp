;;;; ad/rules-batch-elementwise: 要素ごとのプリミティブのバッチ化ルール
;;;; （issue #125 は add だけ。#128 で add sub mul div neg exp log tanh max min
;;;; compare select convert stop-gradient に揃えた）。
;;;;
;;;; JAX の broadcast_batcher に倣った共通ルール1つ（%BATCH-ELEMENTWISE）で済ませ、
;;;; プリミティブごとの個別のルールは書かない。

(in-package #:nabla)

(defun %elementwise-common-axis (batch-dims)
  "BATCH-DIMS（各引数のバッチ軸か NIL）から、揃える先の軸を決める。バッチされた引数が
すべて同じ軸ならその軸（transpose が要らない）、そうでなければ先頭（0）。"
  (let ((dims (remove nil batch-dims)))
    (if (every (lambda (d) (= d (first dims))) dims)
        (first dims)
        0)))

(defun %batch-elementwise (name args batch-dims params)
  "形の揃った要素演算 NAME（PARAMS はそのプリミティブのパラメータの plist）のバッチ化。
バッチされた引数のバッチ軸を共通の位置へ動かし（動かす必要があるものだけ transpose）、
バッチされていない引数は、その位置に長さ SIZE の軸を足して（broadcast-in-dim）形を揃えてから、
元のプリミティブを1つ適用する。要素演算は形が全引数で一致する前提（暗黙の rank 0 の
broadcast は無い）なので、バッチされていない引数は元の形のまま来る。
単一出力として (values (list out) (list axis)) を返す。"
  (unless (some #'identity batch-dims)
    (error 'vmap-error
           :format-control "~S のバッチ化ルールにバッチされた引数が1つも無い（変換側が短絡するはず）"
           :format-arguments (list name)))
  (let* ((axis (%elementwise-common-axis batch-dims))
         (size (loop for arg in args for dim in batch-dims
                     when dim return (nth dim (aval-shape (tracer-aval arg)))))
         (aligned (loop for arg in args for dim in batch-dims
                        collect (if dim
                                    (%vmap-move-axis arg dim axis)
                                    (%vmap-broadcast-batch arg axis size)))))
    (values (list (apply #'%trace-eqn name aligned params)) (list axis))))

(def-batch-rule add (args batch-dims) (%batch-elementwise :add args batch-dims nil))
(def-batch-rule sub (args batch-dims) (%batch-elementwise :sub args batch-dims nil))
(def-batch-rule mul (args batch-dims) (%batch-elementwise :mul args batch-dims nil))
(def-batch-rule div (args batch-dims) (%batch-elementwise :div args batch-dims nil))
(def-batch-rule max (args batch-dims) (%batch-elementwise :max args batch-dims nil))
(def-batch-rule min (args batch-dims) (%batch-elementwise :min args batch-dims nil))
(def-batch-rule neg (args batch-dims) (%batch-elementwise :neg args batch-dims nil))
(def-batch-rule exp (args batch-dims) (%batch-elementwise :exp args batch-dims nil))
(def-batch-rule log (args batch-dims) (%batch-elementwise :log args batch-dims nil))
(def-batch-rule tanh (args batch-dims) (%batch-elementwise :tanh args batch-dims nil))
(def-batch-rule select (args batch-dims) (%batch-elementwise :select args batch-dims nil))
(def-batch-rule stop-gradient (args batch-dims)
  (%batch-elementwise :stop-gradient args batch-dims nil))
(def-batch-rule compare (args batch-dims &key direction)
  (%batch-elementwise :compare args batch-dims (list :direction direction)))
(def-batch-rule convert (args batch-dims &key dtype)
  (%batch-elementwise :convert args batch-dims (list :dtype dtype)))
