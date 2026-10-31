;;;; nabla/data のパッケージ（フェーズ5、issue #278 / #279）。
;;;;
;;;; データ供給の層（Grain 相当）を置くシステム。最初に入るのは data-source
;;;; （インデックスでレコードを取れる抽象）と、メモリ上の配列から作る array-source。
;;;;
;;;; 依存の方針（issue #279）: nabla のシステムのうち core（nabla）にだけ依存する。
;;;; レコードとバッチは jit に渡す PyTree そのものなので、木の扱い（tree-flatten /
;;;; defpytree）は core のものを使い、data 層で別の定義を持たない。nabla/nn と
;;;; 実行系（backend）を実装するシステムには依存しない（FFI を呼ばない純 Lisp の層に
;;;; 保つ。core と同じく、src/ の下で実行系の名前を出さない）。

(defpackage #:nabla.data
  (:use #:cl)
  (:export
   ;; コンディション
   #:data-error
   #:data-source-error
   #:data-source-error-reason
   #:data-source-error-detail
   ;; data-source のプロトコル（issue #279）
   #:data-source
   #:source-length
   #:source-item
   ;; メモリ上の配列のデータソース
   #:array-source
   #:make-array-source
   ;; インデックスのサンプラー（issue #280）
   #:index-sampler
   #:make-index-sampler
   #:sampler-index
   #:sampler-error
   #:sampler-error-reason
   #:sampler-error-detail
   ;; map / filter の変換（issue #281）
   #:map-transform
   #:filter-transform
   #:apply-transforms
   ;; レコードをバッチに積む（issue #281）
   #:batch-records
   #:batch-error
   #:batch-error-reason
   #:batch-error-detail
   ;; ローダとイテレータ（issue #285）
   #:data-loader
   #:data-loader-error
   #:data-loader-error-reason
   #:data-loader-error-detail
   #:data-iterator
   #:make-data-iterator
   #:next-batch
   #:epoch
   ;; 再開状態（issue #285）
   #:iterator-state
   #:iterator-state-key
   #:iterator-state-epoch
   #:iterator-state-position
   #:iterator-state-version
   #:restore-iterator)
  (:documentation
   "nabla/data: データ供給の層。data-source（インデックスで引けるデータ）と array-source、
index-sampler（位置からレコードのインデックスを決めるサンプラー）、map / filter の変換、
data-loader（バッチを作るローダ）と data-iterator、その再開状態 iterator-state。"))
