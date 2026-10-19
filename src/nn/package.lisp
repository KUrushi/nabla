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
   #:safetensors-error-detail
   ;; defmodule（issue #202）
   #:defmodule
   #:call
   #:split-keys
   #:map-state
   #:module-slots
   ;; チェックポイント（issue #206）
   #:save-checkpoint
   #:load-checkpoint
   #:tree-paths
   #:checkpoint-error
   #:checkpoint-error-reason
   #:checkpoint-error-name
   #:checkpoint-error-detail
   ;; 基本レイヤー（issue #204）。make-* はパラメータを与えて作る（チェックポイントの読み込み・テスト用）
   #:dense #:make-dense #:init-dense #:dense-kernel #:dense-bias
   #:layer-norm #:make-layer-norm #:init-layer-norm
   #:layer-norm-scale #:layer-norm-bias #:layer-norm-epsilon
   #:embedding #:make-embedding #:init-embedding #:embedding-embedding
   #:dropout #:make-dropout #:init-dropout #:dropout-rate
   #:sequential #:make-sequential #:init-sequential #:sequential-layers
   #:activation #:make-activation #:init-activation #:activation-fn
   #:relu #:gelu #:softmax #:log-softmax
   ;; オプティマイザ（issue #205）
   #:defoptimizer
   #:optimizer
   #:optimizer-init
   #:optimizer-update
   #:apply-updates
   #:chain
   #:sgd
   #:adam
   #:adamw
   #:clip-by-global-norm
   #:scale
   #:scale-by-adam
   #:scale-by-schedule
   #:momentum-trace
   #:add-decayed-weights
   #:constant-schedule
   #:linear-schedule
   #:cosine-decay-schedule
   #:warmup-cosine-decay-schedule
   ;; dense-general と attention のマスクの部品（issue #208）
   #:dense-general #:make-dense-general #:init-dense-general #:dense-general-kernel #:dense-general-bias
   #:dense-general-contract-axes
   #:make-causal-mask #:make-attention-mask #:combine-masks
   ;; multi-head attention（issue #208）
   #:multi-head-attention #:make-multi-head-attention #:init-multi-head-attention
   #:multi-head-attention-query #:multi-head-attention-key #:multi-head-attention-value
   #:multi-head-attention-out #:multi-head-attention-dropout-rate)
  (:documentation
   "nabla/nn: モデル記述の層。safetensors の読み書き、defmodule、PyTree のチェックポイント、基本レイヤー、オプティマイザ、dense-general、multi-head attention。"))
