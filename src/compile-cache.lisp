;;;; compile-cache: BACKEND-COMPILE の結果をディスクにキャッシュする
;;;; （vmfb ディスクキャッシュ、issue #10）。
;;;;
;;;; BACKEND そのものは変えず、BACKEND-COMPILE に :AROUND メソッドを足す形で
;;;; 実装する（src/backend.lisp の BACKEND-COMPILE の docstring 参照）。
;;;; core は実行系の実装を知らない設計（issue #9）なので、ここでは
;;;; BACKEND-FINGERPRINT が返す文字列のリストと TEXT だけを見て、実際に
;;;; 何をコンパイルしているかは一切気にしない。

(in-package #:nabla)

(defvar *compile-cache-directory* :default
  "BACKEND-COMPILE の :AROUND キャッシュが読み書きするディレクトリ。

呼び出しのたびに解決する（メモ化しない）ので、テストの途中で束縛し
直せる。

- :DEFAULT（既定）: 環境変数 NABLA_CACHE_DIR があれば <それ>/vmfb/、
  無ければ (uiop:xdg-cache-home \"nabla/vmfb/\")
  （= ${XDG_CACHE_HOME:-~/.cache}/nabla/vmfb/）
- pathname または文字列: そのディレクトリをそのまま使う
- NIL: キャッシュを無効にする（毎回 BACKEND-COMPILE を呼び、何も書かない）")

(defparameter +compile-cache-magic+ "NBLMOD01"
  "キャッシュファイルの先頭8バイト（ASCII）。フォーマットの版を兼ねる。

DEFCONSTANT ではなく DEFPARAMETER にしているのは、文字列は EQL で
比較されないため、DEFCONSTANT のまま再コンパイル（ファイルを直して
fasl を作り直す）すると SBCL が
\"defconstant ... uneql to the previous value\" を signal するため。")

(defconstant +compile-cache-header-length+ 40
  "マジック（8バイト）+ payload の SHA-256（32バイト）の長さ。")

(defun %compile-cache-root ()
  "*COMPILE-CACHE-DIRECTORY* を実際のキャッシュディレクトリの pathname に
解決する。NIL ならキャッシュ無効を表す NIL をそのまま返す。"
  (let ((value *compile-cache-directory*))
    (cond
      ((null value) nil)
      ((eq value :default)
       (let ((env (uiop:getenv "NABLA_CACHE_DIR")))
         (if (and env (plusp (length env)))
             (uiop:ensure-directory-pathname
              (merge-pathnames "vmfb/" (uiop:ensure-directory-pathname env)))
             (uiop:xdg-cache-home "nabla/vmfb/"))))
      (t (uiop:ensure-directory-pathname value)))))

(defun %compile-cache-key (backend text)
  "BACKEND-FINGERPRINT の各文字列 + TEXT を、それぞれ「UTF-8 バイト長:」の
ASCII 接頭辞つきで SHA-256 に流し込んだ digest の16進文字列を返す。

長さ接頭辞を挟むのは、文字列を単純に連結すると
'(\"ab\" \"c\")' と '(\"a\" \"bc\")' のようにキーが衝突しうるため
（連結の曖昧さを消す）。"
  (let ((digest (ironclad:make-digest :sha256)))
    (dolist (string (append (backend-fingerprint backend) (list text)))
      (let ((octets (sb-ext:string-to-octets string :external-format :utf-8)))
        (ironclad:update-digest
         digest (sb-ext:string-to-octets (format nil "~D:" (length octets))
                                          :external-format :ascii))
        (ironclad:update-digest digest octets)))
    (ironclad:byte-array-to-hex-string (ironclad:produce-digest digest))))

(defun %compile-cache-path (directory key)
  (merge-pathnames (format nil "~A.module" key) directory))

(defun %read-octets (path)
  "PATH の中身を丸ごと (SIMPLE-ARRAY (UNSIGNED-BYTE 8) (*)) にして返す。"
  (with-open-file (stream path :direction :input :element-type '(unsigned-byte 8))
    (let* ((length (file-length stream))
           (octets (make-array length :element-type '(unsigned-byte 8))))
      (read-sequence octets stream)
      octets)))

(defun %compile-cache-read (directory key)
  "DIRECTORY の中の KEY に対応するキャッシュファイルを読む。

有効なら (VALUES T payload-octets) を返す。ファイルが無い・
+COMPILE-CACHE-HEADER-LENGTH+ バイト未満・マジック不一致・payload の
SHA-256 が記録された digest と不一致・読み込み中に何らかのエラー、の
どれかなら壊れている（または存在しない）とみなし、あれば
IGNORE-ERRORS で静かに削除してから (VALUES NIL NIL) を返す（次回また
壊れたファイルを読まないようにするため）。"
  (let ((path (%compile-cache-path directory key)))
    (handler-case
        (let ((octets (%read-octets path)))
          (if (< (length octets) +compile-cache-header-length+)
              (progn (ignore-errors (delete-file path)) (values nil nil))
              (let* ((magic (sb-ext:octets-to-string
                             (subseq octets 0 8) :external-format :ascii))
                     (stored-digest (subseq octets 8 +compile-cache-header-length+))
                     (payload (subseq octets +compile-cache-header-length+)))
                (if (and (string= magic +compile-cache-magic+)
                         (equalp stored-digest
                                 (ironclad:digest-sequence :sha256 payload)))
                    (values t payload)
                    (progn (ignore-errors (delete-file path)) (values nil nil))))))
      (error ()
        (ignore-errors (delete-file path))
        (values nil nil)))))

(defun %compile-cache-header (payload)
  "マジック（8バイト）+ PAYLOAD の SHA-256（32バイト、生のバイト列）を
連結した +COMPILE-CACHE-HEADER-LENGTH+ バイトのヘッダーを返す。

CONCATENATE の結果型に (SIMPLE-ARRAY (UNSIGNED-BYTE 8) (*)) のような
'*' を含む型指定子を書くと、tools/mutate の算術演算子の入れ替え変異が
その '*' を '/' と誤認して書き換え、SBCL の型パーサがクラッシュする
（'*' は次元がいくつでもよいという型指定子の記法であって、掛け算では
ない）。REPLACE で組み立てれば、そのような型指定子を書かずに済む。"
  (let ((header (make-array +compile-cache-header-length+ :element-type '(unsigned-byte 8))))
    (replace header (sb-ext:string-to-octets +compile-cache-magic+ :external-format :ascii))
    (replace header (ironclad:digest-sequence :sha256 payload) :start1 8)
    header))

(defun %compile-cache-write (directory key payload)
  "DIRECTORY の中に KEY に対応するキャッシュファイルを、一時ファイルへの
書き込み + rename でアトミックに作る。同じキーで複数スレッドが同時に
書いても、それぞれ完全なファイルを書いた上で最後の rename が勝つだけ
なので、ロックは取らない。"
  (ensure-directories-exist directory)
  (let* ((final-path (%compile-cache-path directory key))
         (tmp-path (merge-pathnames
                    (format nil "~A.~D.tmp" key (random most-positive-fixnum (make-random-state t)))
                    directory))
         (header (%compile-cache-header payload)))
    (with-open-file (stream tmp-path :direction :output
                                      :element-type '(unsigned-byte 8)
                                      :if-exists :supersede
                                      :if-does-not-exist :create)
      (write-sequence header stream)
      (write-sequence payload stream)
      (finish-output stream))
    (uiop:rename-file-overwriting-target tmp-path final-path)
    final-path))

(defmethod backend-compile :around ((backend backend) text)
  "キャッシュが有効（*COMPILE-CACHE-DIRECTORY* が NIL でない）なら、
BACKEND-FINGERPRINT + TEXT から求めたキーでディスクを引く。ヒットすれば
CALL-NEXT-METHOD（実際のコンパイル）を呼ばずにそのバイト列を返す。
ミスなら CALL-NEXT-METHOD の結果をキャッシュに書いてから返す。
CALL-NEXT-METHOD が signal したエラーは、何も書かずにそのまま伝播する。
キャッシュが無効なら常に CALL-NEXT-METHOD をそのまま呼ぶ。"
  (let ((directory (%compile-cache-root)))
    (if (null directory)
        (call-next-method)
        (let ((key (%compile-cache-key backend text)))
          (multiple-value-bind (hit payload) (%compile-cache-read directory key)
            (if hit
                payload
                (let ((result (call-next-method)))
                  (%compile-cache-write directory key result)
                  result)))))))
