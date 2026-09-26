;;;; StableHLO のテキストを IREE でコンパイルして vmfb のバイト列を得る、
;;;; nabla.iree の公開 API。
;;;;
;;;; アルゴリズム（すべて $NABLA_IREE_HOME/include/iree/compiler/embedding_api.h
;;;; の埋め込み C API どおり）:
;;;;
;;;;   ensure-compiler-loaded
;;;;   -> SessionCreate -> SessionSetFlags（失敗なら phase :flags）
;;;;   -> InvocationCreate -> EnableCallbackDiagnostics（診断を集める）
;;;;   -> SourceWrapBuffer -> ParseSource（失敗なら phase :parse）
;;;;   -> Pipeline（失敗なら phase :compile）
;;;;   -> OutputOpenMembuffer -> OutputVMBytecode（失敗なら phase :output）
;;;;   -> OutputMapMemory -> バイト列にコピー
;;;;   -> unwind-protect で OutputDestroy, InvocationDestroy,
;;;;      SourceDestroy, SessionDestroy の順に後始末
;;;;
;;;; セッションと invocation は呼び出しごとに新しく作る（セッション自体は
;;;; スレッドセーフではないため、これが compile-stablehlo をスレッドセーフに
;;;; している）。
;;;;
;;;; 注意（issue #5 で踏んだフレーキーなクラッシュ）: このプロセスで一度でも
;;;; compile-stablehlo（正確には ireeCompilerInvocationPipeline）を実行すると、
;;;; IREE/LLVM/MLIR は最適化パス用の永続的なワーカースレッドプール
;;;; （"llvm-worker-N"、SBCL の管理外のスレッド）を遅延生成し、以後プロセスが
;;;; 終わるまで生かしたままにする。この状態で SB-EXT:GC を明示的に呼ぶと
;;;; （:full の有無に関わらず）、SBCL 2.2.9（safepoint 無しビルド）の GC が
;;;; "no SP known for thread" という致命的エラーで確実に落ちることを確認して
;;;; いる。試した緩和策（--mlir-disable-threading 相当のセッションフラグ
;;;; ―― このバージョンの embedding API には存在しない、呼び出しスレッドの
;;;; シグナルを全部ブロックする、専用の SBCL スレッドに呼び出しを移す、
;;;; dlmopen で別のリンクマップ名前空間に読み込む――はどれも効果がなかった。
;;;; 一方、確保のしきい値で自動的に走る通常の GC（明示的に SB-EXT:GC を呼ば
;;;; ない、nabla の実際のコード・テストが使う経路）は、malformed/valid を
;;;; 混ぜて1000回以上コンパイルしながら確保し続けても再現しなかった
;;;; （tests/iree/compiler-test.lisp の organic-gc-pressure テスト参照）。
;;;; そのため、nabla.iree をロードしたプロセスでは SB-EXT:GC（および
;;;; TRIVIAL-GARBAGE:GC）を明示的に呼ばないこと。これは SBCL の非 safepoint
;;;; スレッド実装と IREE の永続ワーカースレッドプールの相互作用に起因する
;;;; 既知の制約で、nabla 側のコードのバグではない。

(in-package #:nabla.iree)

(defun compiler-api-version ()
  "ireeCompilerGetAPIVersion の結果を (values major minor) にして返す。
上位16ビットがメジャー、下位16ビットがマイナー。"
  (ensure-compiler-loaded)
  (let ((raw (%compiler-get-api-version)))
    (values (ash raw -16) (logand raw #xFFFF))))

(defun compiler-revision ()
  "IREE コンパイラのビルドリビジョン文字列を返す（ireeCompilerGetRevision）。"
  (ensure-compiler-loaded)
  (or (%compiler-get-revision) ""))

(defun compile-flags (target &key cuda-arch)
  "TARGET（:local または :cuda）向けの iree-compile 相当のフラグをリストで返す。
:local は CLAUDE.md / verify-iree.sh と同じ CPU 向けのレシピ
（llvm-cpu、target-cpu=host。生成される vmfb はこのため実行するマシンに
依存する）。:cuda は CUDA-ARCH（例: \"sm_80\"）を渡すと
--iree-cuda-target=CUDA-ARCH を追加する。TARGET がこれ以外なら型エラーを
signal する。呼び出しごとに新しいリストを作るが、内容は決定的。"
  (ecase target
    (:local (list "--iree-input-type=stablehlo"
                   "--iree-hal-target-device=local"
                   "--iree-hal-local-target-device-backends=llvm-cpu"
                   "--iree-llvmcpu-target-cpu=host"))
    (:cuda (append (list "--iree-input-type=stablehlo"
                          "--iree-hal-target-device=cuda")
                    (when cuda-arch
                      (list (format nil "--iree-cuda-target=~A" cuda-arch)))))))

;; ParseSource / Pipeline の失敗中に集める診断。callback は「invocation の
;; 破棄までどのスレッドからでも」呼ばれうる (embedding_api.h) ので、実行中の
;; compile-stablehlo 呼び出しを、動的束縛ではなく整数のクッキーで識別し、
;; ミューテックスで守ったハッシュ表に集める。

(defvar *diagnostics-lock* (sb-thread:make-mutex :name "nabla-iree-diagnostics"))
(defvar *diagnostics-table* (make-hash-table)
  "クッキー（整数）-> 集めている診断のリスト（逆順）。")
(defvar *next-diagnostics-cookie* 0)

(defun %diagnostics-begin ()
  (sb-thread:with-mutex (*diagnostics-lock*)
    (let ((cookie (incf *next-diagnostics-cookie*)))
      (setf (gethash cookie *diagnostics-table*) nil)
      cookie)))

(defun %diagnostics-push (cookie severity text)
  (sb-thread:with-mutex (*diagnostics-lock*)
    (push (cons severity text) (gethash cookie *diagnostics-table*))))

(defun %diagnostics-end (cookie)
  (sb-thread:with-mutex (*diagnostics-lock*)
    (prog1 (nreverse (gethash cookie *diagnostics-table*))
      (remhash cookie *diagnostics-table*))))

(cffi:defcallback %diagnostic-callback :void
    ((severity :int) (message :pointer) (message-size :size) (user-data :pointer))
  (let ((cookie (cffi:pointer-address user-data))
        (text (cffi:foreign-string-to-lisp message :count message-size :encoding :utf-8)))
    (%diagnostics-push cookie (%diagnostic-severity-keyword severity) text)))

(defun %copy-membuffer-to-octets (contents size)
  "CONTENTS（foreign :pointer）から SIZE バイトを読み、新しい
(simple-array (unsigned-byte 8) (*)) にコピーして返す。vmfb は数 MB になり
うるので、1バイトずつ mem-aref する代わりに memcpy を1回呼ぶ。"
  (let ((bytes (make-array size :element-type '(unsigned-byte 8))))
    (sb-sys:with-pinned-objects (bytes)
      (cffi:foreign-funcall "memcpy"
                             :pointer (sb-sys:vector-sap bytes)
                             :pointer contents
                             :size size
                             :pointer))
    bytes))

(defun %session-set-flags (session flags)
  (let ((argc (length flags)))
    (cffi:with-foreign-object (argv :pointer argc)
      (let ((c-strings nil))
        (unwind-protect
             (progn
               (loop for flag in flags
                     for i from 0
                     do (let ((c-string (cffi:foreign-string-alloc flag)))
                          (push c-string c-strings)
                          (setf (cffi:mem-aref argv :pointer i) c-string)))
               (let ((error (%compiler-session-set-flags session argc argv)))
                 (unless (cffi:null-pointer-p error)
                   (let ((message (%compiler-error-get-message error)))
                     (%compiler-error-destroy error)
                     (error 'iree-compile-error :phase :flags :message message)))))
          (dolist (c-string c-strings)
            (cffi:foreign-string-free c-string)))))))

(defun compile-stablehlo (text &key (flags (compile-flags :local)) (source-name "nabla.mlir"))
  "StableHLO の TEXT を IREE でコンパイルし、vmfb のバイト列を
(simple-array (unsigned-byte 8) (*)) として返す。FLAGS は iree-compile 相当の
コマンドライン引数のリスト（既定は (compile-flags :local)）。

コンパイルが失敗すると IREE-COMPILE-ERROR を signal する。フラグの設定・
構文解析・パイプラインの実行・出力のどの段階で失敗したかが PHASE に、
MLIR の診断（あれば）が DIAGNOSTICS に入る。共有ライブラリが見つからない
ときは IREE-LIBRARY-NOT-FOUND を signal する。

呼び出しごとに新しいセッションと invocation を作るので、複数スレッドから
並行に呼んでよい。"
  (ensure-compiler-loaded)
  (let (session invocation source output)
    (unwind-protect
         (progn
           (setf session (%compiler-session-create))
           (%session-set-flags session flags)
           (setf invocation (%compiler-invocation-create session))
           (let ((cookie (%diagnostics-begin)))
             (unwind-protect
                  ;; ireeCompilerSourceWrapBuffer は TEXT のバイト列をコピーせず、
                  ;; 渡したバッファをそのまま「包む」だけ（ヘッダの言う「ソースの
                  ;; 処理が終わるまでソースを生かしておく」の裏側）。そのため
                  ;; ParseSource（と、安全のためそれ以降の処理すべて）を
                  ;; with-foreign-string の動的エクステントの外に出してはいけない
                  ;; ——外に出すと、解放済みのメモリを読むことになり、無関係な
                  ;; 文字化けとして構文エラーが出る（実際に踏んだバグ）。
                  (cffi:with-foreign-string ((buffer length) text :encoding :utf-8)
                    (%compiler-invocation-enable-callback-diagnostics
                     invocation 0 (cffi:callback %diagnostic-callback) (cffi:make-pointer cookie))
                    (cffi:with-foreign-object (out-source :pointer)
                      (let ((error (%compiler-source-wrap-buffer
                                    session source-name buffer length t out-source)))
                        (unless (cffi:null-pointer-p error)
                          (let ((message (%compiler-error-get-message error)))
                            (%compiler-error-destroy error)
                            (error 'iree-compile-error :phase :parse :message message
                                                        :diagnostics (%diagnostics-end cookie)))))
                      (setf source (cffi:mem-ref out-source :pointer)))
                    (unless (%compiler-invocation-parse-source invocation source)
                      (error 'iree-compile-error :phase :parse
                                                  :diagnostics (%diagnostics-end cookie)))
                    (unless (%compiler-invocation-pipeline invocation +compiler-pipeline-std+)
                      (error 'iree-compile-error :phase :compile
                                                  :diagnostics (%diagnostics-end cookie)))
                    (cffi:with-foreign-object (out-output :pointer)
                      (let ((error (%compiler-output-open-membuffer out-output)))
                        (unless (cffi:null-pointer-p error)
                          (let ((message (%compiler-error-get-message error)))
                            (%compiler-error-destroy error)
                            (error 'iree-compile-error :phase :output :message message
                                                        :diagnostics (%diagnostics-end cookie)))))
                      (setf output (cffi:mem-ref out-output :pointer)))
                    (let ((error (%compiler-invocation-output-vm-bytecode invocation output)))
                      (unless (cffi:null-pointer-p error)
                        (let ((message (%compiler-error-get-message error)))
                          (%compiler-error-destroy error)
                          (error 'iree-compile-error :phase :output :message message
                                                      :diagnostics (%diagnostics-end cookie)))))
                    (cffi:with-foreign-objects ((out-contents :pointer) (out-size :uint64))
                      (let ((error (%compiler-output-map-memory output out-contents out-size)))
                        (unless (cffi:null-pointer-p error)
                          (let ((message (%compiler-error-get-message error)))
                            (%compiler-error-destroy error)
                            (error 'iree-compile-error :phase :output :message message
                                                        :diagnostics (%diagnostics-end cookie)))))
                      (%copy-membuffer-to-octets (cffi:mem-ref out-contents :pointer)
                                                  (cffi:mem-ref out-size :uint64))))
               ;; 正常終了時のクッキーの後始末（例外経路では上の各所ですでに
               ;; %diagnostics-end 済みなので二重に消しても実害はない）。
               (sb-thread:with-mutex (*diagnostics-lock*)
                 (remhash cookie *diagnostics-table*)))))
      (when output (%compiler-output-destroy output))
      (when invocation (%compiler-invocation-destroy invocation))
      (when source (%compiler-source-destroy source))
      (when session (%compiler-session-destroy session)))))
