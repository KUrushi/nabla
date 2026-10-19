;;;; nabla/nn のパッケージ（issue #198）。
;;;;
;;;; モデル記述の層（defmodule・レイヤー・オプティマイザ・チェックポイント）を
;;;; 置くシステム。最初に入るのは safetensors の読み書き（平坦な名前 → 配列）。

(defpackage #:nabla.nn
  (:use #:cl)
  (:export
   #:encode-safetensors
   #:decode-safetensors
   #:write-safetensors
   #:read-safetensors
   #:safetensors-error
   #:safetensors-error-path
   #:safetensors-error-reason
   #:safetensors-error-detail)
  (:documentation
   "nabla/nn: モデル記述の層。現在は safetensors の読み書きだけ。"))
