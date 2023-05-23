;;;; Some simple stuff I often end up copy->pasting around

(defun zip (&rest lists)
  "Zip list elements together"
  (apply #'mapcar #'list lists))

(defun pairwise (list)
  "Return a list with pairs of adjacent elements from LIST"
  (zip (butlast list) (rest list)))

(defun takewhile (pred list)
  "Collect elements from LIST while PRED is truthy"
  (loop for e in list
        while (funcall pred e)
        collect e))

(defun range (start stop)
  "return a list of integers from the range [START, STOP)"
  (loop for n from start below stop collect n))

(defun randint (start end)
  "Return a random integer in the range [START, END]"
  ;; (random x) => [0, x)
  ;; dist ::= end - start                ; distance between `start` and `end`
  ;; (random (- end start) => [0, dist)  ; excluding the upper limit
  ;; (random (+ 1 (- end start))) => [0, dist + 1) == [0, dist]
  ;; (+ start (random (+ 1 (- end start)))) => [start, end]
  (+ start (random (+ 1 (- end start)))))

;;; Macros from Paul Graham's book "On Lisp"
;;; As see in the section 11.4, by defining these macros to expand DOs, they
;;; inherit the DO's RETURN capability.
(defmacro while (test &body body)
  "Repeat BODY as long as TEST is truthy"
  `(do ()
    ((not ,test))
    ,@body))

(defmacro till (test &body body)
  "Repeat BODY as long as TEST is falsey"
  `(do ()
    (,test)
    ,@body))

(defmacro for ((var start stop) &body body)
  "Repeat BODY with VAR set from the range [START, STOP)"
  (let ((gstop (gensym)))
    `(do ((,var ,start (1+ ,var))
          (,gstop ,stop))
      ((> ,var ,gstop))
      ,@body)))
