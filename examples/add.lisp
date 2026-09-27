(require :asdf)
(asdf:load-system "nabla/iree")
(defparameter *stablehlo* "
func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %a, %b : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}")
(let* ((backend (nb:find-backend :iree))                       ; プロセスに1つの IREE backend（CPU）
       (module (nb:backend-load backend (nb:backend-compile backend *stablehlo*)))
       (a (nb:to-device (make-array 4 :element-type 'single-float :initial-contents '(1.0 2.0 3.0 4.0)) backend))
       (b (nb:to-device (make-array 4 :element-type 'single-float :initial-contents '(10.0 20.0 30.0 40.0)) backend))
       (result (nb:backend-invoke backend module "main" a b)))
  (format t "~&~A~%~A~%" (nb:device-array-aval result) (nb:to-host result))
  (nb:backend-unload backend module))
