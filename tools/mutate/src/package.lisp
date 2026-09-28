;;;; package.lisp -- nabla-mutate のパッケージ定義

(defpackage #:nabla.mutate
  (:use #:cl)
  (:nicknames #:nb.mutate)
  (:export
   ;; runner
   #:run
   #:plan-mutants
   #:report
   #:report-mutants
   #:mutation-score
   #:mutant
   #:mutant-file
   #:mutant-line
   #:mutant-original-form
   #:mutant-mutated-form
   #:mutant-operator
   #:mutant-status
   #:default-exclusions-path
   #:default-test-function
   #:*regression-directories*
   ;; reader
   #:missing-source-file
   #:source-form
   #:make-source-form
   #:source-form-form
   #:source-form-start-line
   #:source-form-end-line
   #:source-form-package
   #:read-source-forms
   #:definitions-in-range
   #:mutable-definition-p
   ;; mutate
   #:mutate-form
   #:mutation-sites
   #:arid-node-p
   #:*mutation-operators*
   ;; diff
   #:ranges-from-git-diff
   ;; exclusions
   #:load-exclusions
   #:excluded-p))
