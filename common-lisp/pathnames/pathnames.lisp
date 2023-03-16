;;;; pathnames.lisp

(in-package #:pathnames)

(defun component-present-p (component)
  "Test if a pathname component is not NIL or :UNSPECIFIC."
  (and component
       (not (eql component :unspecific))
       component))

(defun directory-pathname-p (spec)
  "Check if SPEC (a pathname designator) is a valid directory"
  (and (not (component-present-p (pathname-name spec)))
       (not (component-present-p (pathname-type spec)))
       spec))

(defun pathname-as-directory (spec)
  "Convert a pathname designator into a directory pathname."
  (let ((pname (pathname spec)))
    (when (wild-pathname-p pname)
      (error "Can't reliably convert wild pathnames."))
    (if (not (directory-pathname-p pname))
        (make-pathname
         :directory  (append (or (pathname-directory pname)
                                 (list :relative))
                             (list (file-namestring pname)))
         :name       nil
         :type       nil
         :defaults   pname)
        pname)))

(defun directory-wildcard (spec)
  "Create a wildcard pathname from SPEC."
  (make-pathname
   :defaults (pathname-as-directory spec)
   :name :wild
   :type :wild))

(defun file-exists-p (spec)
  "Check if a spec is an existing file.

If SPEC (a pathname designator) is the path to a file, return its TRUENAME.
Otherwise, return NIL.

Example:
  (file-exists-p \"path/to/file\")          => #P\"truepath/to/file\"
  (file-exists-p \"path/to/file/\")         => NIL
  (file-exists-p \"path/to/missing-file\")  => NIL"
  (let ((probe (probe-file spec)))
    (and probe
         (pathname-name spec)
         (pathname-name probe)
         probe)))

(defun directory-exists-p (spec)
  "Check if a spec is an existing directory.

If SPEC (a pathname designator) is the path to a directory, return its TRUENAME.
Otherwise return NIL. Note that the trailing slash is optional.

Example:
  (directory-exists-p \"path/to/dir\")   => #P\"/truepath/to/dir/\"
  (directory-exists-p \"path/to/dir/\")  => #P\"/truepath/to/dir/\"
  (directory-exists-p \"path/to/file\")  => NIL
  (directory-exists-p \"path/to/file/\") => NIL"
  (let ((probe (probe-file spec)))
    (and probe (not (pathname-name probe)) probe)))

(defun path-exists-p (spec)
  "Check if SPEC (a pathname designator) exists

This function emulates the behaviour of bash's test for
files and directories with its '-f' and '-d' options.

Example: suppose we have a file 'file.ext' and a directory 'dir'
  (path-exists-p \"file.ext\")  => #P\"truepath/to/file.ext\"
  (path-exists-p \"file.ext/\") => NIL
  (path-exists-p \"dir\")       => #P\"truepath/to/dir/\"
  (path-exists-p \"dir/\")      => #P\"truepath/to/dir/\""
  (or (file-exists-p spec) (directory-exists-p spec)))

(defun list-directory (&optional (spec "./"))
  "List file contents from SPEC

Return a list of pathname designators from SPEC, a directory pathname
designator.

Example: Suppose the directory '/home/user/test' has the following contents:

  /home/user/test/
  ├── directory/
  │   └── innerfile/
  ├── file1.txt
  ├── file2.txt
  └── otherfile

then, calling LIST-DIRECTORY on '/home/user/test' yields:

  (list-directory \"/home/user/test\")
      => (#P\"/home/user/test/directory/\"
          #P\"/home/user/test/file1.txt\"
          #P\"/home/user/test/file2.txt\"
          #P\"/home/user/test/otherfile\")"
  (when (wild-pathname-p spec)
    (error "Can only list concrete directory names."))
  ;; Added error condition to differentiate empty and
  ;; non-existing directory paths.
  (unless (directory-exists-p spec)
    (error "No such directory: '~a'" spec))
  (let ((wildcard (directory-wildcard spec)))
    (directory wildcard)))

(defun walk-directory (spec fn &key dirs (test (constantly t)))
  "Recursively apply a function to a directory's contents.

Walk through SPEC (a pathname designator) and apply the
function FN to its contents.

Keywords:
  If :DIRS is not NIL, also apply the function FN to
  directories.

  :TEST can be set to specify the test predicate to filter
    which files should the function FN be applied to. "
  (labels
      ((walk (name)
         (cond ((directory-pathname-p name)
                (and dirs (funcall test name) (funcall fn name))
                (dolist (x (list-directory name)) (walk x)))
               ((funcall test name) (funcall fn name)))))
    (walk (pathname-as-directory spec))))

(defun dirname (spec)
  "Get a path's directory components.

Example:
  (dirname \"path/to/some/name\"  => #P\"path/to/some/\"
  (dirname \"path/to/some/name/\" => #P\"path/to/some/\""
  (let* ((directory (pathname-directory spec))
         (num-components (length directory)))
    (cond
      ((= num-components 0)
       (make-pathname :directory '(:relative ".")))
      ((= num-components 1)
       (make-pathname :directory directory))
      ((directory-pathname-p spec)
       (make-pathname :directory (butlast directory)))
      (t
       (make-pathname :directory directory)))))

(defun basename (spec)
  "Get a path's name and extension.

Example:
  (basename \"/path/to/some/name\") => #P\"name\""
  (let ((name (pathname-name spec))
        (ext (pathname-type spec)))
    (make-pathname :name name :type ext)))
