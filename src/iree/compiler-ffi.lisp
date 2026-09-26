;;;; IREE コンパイラの埋め込み C API の生の CFFI バインディング。
;;;;
;;;; 関数名・シグネチャは $NABLA_IREE_HOME/include/iree/compiler/embedding_api.h
;;;; （IREE v3.11.0, commit e4a3b0405d7d23554da26403658d0e8c3c5ecf25。
;;;; third_party/iree.lock で固定）から書き写した。全ての引数・返り値が
;;;; ポインタ・整数・bool・size_t なので、構造体を値渡しする必要がなく、
;;;; 素の CFFI（cffi-libffi は不要）で足りる。
;;;;
;;;; ireeCompilerLoadLibrary は libIREECompiler.so から export されていない
;;;; （compiler/bindings/c/iree/compiler/loader/loader.cpp の静的シムが実装
;;;; していて、これを dlopen する側が自前で用意する関数）。代わりに
;;;; cffi:load-foreign-library（library.lisp）が dlopen そのものを行い、
;;;; ここでは他の ireeCompiler* シンボルを直接 defcfun する。
;;;;
;;;; ここのシンボルは % 接頭辞を付け、export しない。呼び出し側は
;;;; compiler.lisp のラッパー（% の付かない関数）だけを使う。

(in-package #:nabla.iree)

;; embedding_api.h の `bool` は C/C++ の1バイトの bool。CFFI の :boolean は
;; 既定で base-type が :int（4バイト）になり、1バイトしか書き込まれない
;; 返り値の上位バイトが不定になりうる（x86-64 の呼び出し規約は戻り値の
;; 上位ビットを保証しない）。base-type を明示的に :int8 にして、実際の
;; C の bool と幅を合わせる。
(cffi:defctype %bool (:boolean :int8))

;; エラー。

(cffi:defcfun ("ireeCompilerErrorDestroy" %compiler-error-destroy) :void
  (error :pointer))

(cffi:defcfun ("ireeCompilerErrorGetMessage" %compiler-error-get-message) :string
  (error :pointer))

;; グローバル初期化。

(cffi:defcfun ("ireeCompilerGetAPIVersion" %compiler-get-api-version) :int)

(cffi:defcfun ("ireeCompilerGlobalInitialize" %compiler-global-initialize) :void)

(cffi:defcfun ("ireeCompilerGetRevision" %compiler-get-revision) :string)

;; セッション。

(cffi:defcfun ("ireeCompilerSessionCreate" %compiler-session-create) :pointer)

(cffi:defcfun ("ireeCompilerSessionDestroy" %compiler-session-destroy) :void
  (session :pointer))

(cffi:defcfun ("ireeCompilerSessionSetFlags" %compiler-session-set-flags) :pointer
  (session :pointer)
  (argc :int)
  (argv :pointer))

;; 実行（invocation）。

(cffi:defcfun ("ireeCompilerInvocationCreate" %compiler-invocation-create) :pointer
  (session :pointer))

(cffi:defcfun ("ireeCompilerInvocationEnableCallbackDiagnostics"
               %compiler-invocation-enable-callback-diagnostics)
    :void
  (invocation :pointer)
  (flags :int)
  (callback :pointer)
  (user-data :pointer))

(cffi:defcfun ("ireeCompilerInvocationDestroy" %compiler-invocation-destroy) :void
  (invocation :pointer))

(cffi:defcfun ("ireeCompilerInvocationParseSource" %compiler-invocation-parse-source) %bool
  (invocation :pointer)
  (source :pointer))

(cffi:defcfun ("ireeCompilerInvocationPipeline" %compiler-invocation-pipeline) %bool
  (invocation :pointer)
  (pipeline :int))

(cffi:defcfun ("ireeCompilerInvocationOutputVMBytecode"
               %compiler-invocation-output-vm-bytecode)
    :pointer
  (invocation :pointer)
  (output :pointer))

;; ソース（入力）。

(cffi:defcfun ("ireeCompilerSourceDestroy" %compiler-source-destroy) :void
  (source :pointer))

(cffi:defcfun ("ireeCompilerSourceWrapBuffer" %compiler-source-wrap-buffer) :pointer
  (session :pointer)
  (buffer-name :string)
  (buffer :pointer)
  (length :size)
  (null-terminated %bool)
  (out-source :pointer))

;; 出力。

(cffi:defcfun ("ireeCompilerOutputDestroy" %compiler-output-destroy) :void
  (output :pointer))

(cffi:defcfun ("ireeCompilerOutputOpenMembuffer" %compiler-output-open-membuffer) :pointer
  (out-output :pointer))

(cffi:defcfun ("ireeCompilerOutputMapMemory" %compiler-output-map-memory) :pointer
  (output :pointer)
  (out-contents :pointer)
  (out-size :pointer))

;; iree_compiler_pipeline_t の値（enum、embedding_api.h より）。
(defconstant +compiler-pipeline-std+ 0)

;; iree_compiler_diagnostic_severity_t の値。
(defconstant +diagnostic-severity-note+ 0)
(defconstant +diagnostic-severity-warning+ 1)
(defconstant +diagnostic-severity-error+ 2)
(defconstant +diagnostic-severity-remark+ 3)

(defun %diagnostic-severity-keyword (code)
  (case code
    (#.+diagnostic-severity-note+ :note)
    (#.+diagnostic-severity-warning+ :warning)
    (#.+diagnostic-severity-error+ :error)
    (#.+diagnostic-severity-remark+ :remark)
    (t :unknown)))
