;;;; package.lisp

(defpackage #:pathnames
  (:use #:cl)
  (:nicknames #:path)
  (:export
   :directory-pathname-p
   :pathname-as-directory
   :directory-wildcard
   :list-directory
   :walk-directory
   :file-exists-p
   :directory-exists-p
   :path-exists-p
   :dirname
   :basename))
