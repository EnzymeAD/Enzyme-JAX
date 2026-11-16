// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --drop-unsupported-attributes | stablehlo-translate - --interpret
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s

// When several elements tie with the result of a reduce maximum/minimum, only
// the last one in row-major order over the reduced dimensions gets the
// adjoint. A NaN beats any number, and the init value gets the adjoint only
// when no input element matches. The forward pass caches the winner's index.

func.func @f(%x : tensor<2x3x2xf32>, %init : tensor<f32>) -> tensor<3xf32> {
  %r = stablehlo.reduce(%x init: %init) applies stablehlo.maximum across dimensions = [0, 2] : (tensor<2x3x2xf32>, tensor<f32>) -> tensor<3xf32>
  func.return %r : tensor<3xf32>
}
func.func @g(%x : tensor<2x3xf32>, %init : tensor<f32>) -> tensor<2xf32> {
  %r = stablehlo.reduce(%x init: %init) applies stablehlo.minimum across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
  func.return %r : tensor<2xf32>
}
func.func @main() {
  // window 0: all equal (4 ties) -> last = x[1,0,1]
  // window 1: ties at x[0,1,1] and x[1,1,0] -> last = x[1,1,0]
  // window 2: all below init (5) -> init wins
  %x = stablehlo.constant dense<[[[1.0, 1.0], [2.0, 3.0], [0.0, 1.0]], [[1.0, 1.0], [3.0, 0.0], [2.0, 2.0]]]> : tensor<2x3x2xf32>
  %init = stablehlo.constant dense<0.5> : tensor<f32>
  %init2 = stablehlo.constant dense<5.0> : tensor<f32>
  %seed = stablehlo.constant dense<[1.0, 10.0, 100.0]> : tensor<3xf32>
  %g:2 = enzyme.autodiff @f(%x, %init, %seed) <{activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>]}> : (tensor<2x3x2xf32>, tensor<f32>, tensor<3xf32>) -> (tensor<2x3x2xf32>, tensor<f32>)
  check.expect_almost_eq_const %g#0, dense<[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [10.0, 0.0], [0.0, 100.0]]]> : tensor<2x3x2xf32>
  check.expect_almost_eq_const %g#1, dense<0.0> : tensor<f32>
  %h:2 = enzyme.autodiff @f(%x, %init2, %seed) <{activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>]}> : (tensor<2x3x2xf32>, tensor<f32>, tensor<3xf32>) -> (tensor<2x3x2xf32>, tensor<f32>)
  check.expect_almost_eq_const %h#0, dense<0.0> : tensor<2x3x2xf32>
  check.expect_almost_eq_const %h#1, dense<111.0> : tensor<f32>
  // row 0: ties at x[0,0] and x[0,2] -> last = x[0,2]; row 1: ties with init too
  %y = stablehlo.constant dense<[[1.0, 4.0, 1.0], [2.0, 2.0, 3.0]]> : tensor<2x3xf32>
  %init3 = stablehlo.constant dense<2.0> : tensor<f32>
  %seed2 = stablehlo.constant dense<[1.0, 10.0]> : tensor<2xf32>
  %k:2 = enzyme.autodiff @g(%y, %init3, %seed2) <{activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>]}> : (tensor<2x3xf32>, tensor<f32>, tensor<2xf32>) -> (tensor<2x3xf32>, tensor<f32>)
  check.expect_almost_eq_const %k#0, dense<[[0.0, 0.0, 1.0], [0.0, 10.0, 0.0]]> : tensor<2x3xf32>
  check.expect_almost_eq_const %k#1, dense<0.0> : tensor<f32>
  // row 0: the NaN wins; row 1: last of two NaNs wins
  %z = stablehlo.constant dense<[[1.0, 0x7FC00000, 3.0], [0x7FC00000, 2.0, 0x7FC00000]]> : tensor<2x3xf32>
  %init4 = stablehlo.constant dense<0.0> : tensor<f32>
  %n:2 = enzyme.autodiff @g(%z, %init4, %seed2) <{activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>]}> : (tensor<2x3xf32>, tensor<f32>, tensor<2xf32>) -> (tensor<2x3xf32>, tensor<f32>)
  check.expect_almost_eq_const %n#0, dense<[[0.0, 1.0, 0.0], [0.0, 0.0, 10.0]]> : tensor<2x3xf32>
  check.expect_almost_eq_const %n#1, dense<0.0> : tensor<f32>
  func.return
}

// CHECK-LABEL: func.func private @diffef(
// CHECK-SAME:      %[[X:.+]]: tensor<2x3x2xf32>, %[[INIT:.+]]: tensor<f32>, %[[DRES:.+]]: tensor<3xf32>)
// CHECK-NEXT:    %[[NEG1R:.+]] = stablehlo.constant dense<-1> : tensor<3xi32>
// CHECK-NEXT:    %[[NEG1S:.+]] = stablehlo.constant dense<-1> : tensor<i32>
// CHECK-NEXT:    %[[STRIDE:.+]] = stablehlo.constant dense<2> : tensor<2x3x2xi32>
// CHECK-NEXT:    %[[ZEROR:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x3x2xf32>
// CHECK-NEXT:    %[[ZEROS:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[IOTA2:.+]] = stablehlo.iota dim = 2 : tensor<2x3x2xi32>
// CHECK-NEXT:    %[[IOTA0:.+]] = stablehlo.iota dim = 0 : tensor<2x3x2xi32>
// CHECK-NEXT:    %[[SCALED:.+]] = stablehlo.multiply %[[IOTA0]], %[[STRIDE]]
// CHECK-NEXT:    %[[LIN:.+]] = stablehlo.add %[[IOTA2]], %[[SCALED]]
// CHECK-NEXT:    %[[RED:.+]]:2 = stablehlo.reduce(%[[X]] init: %[[INIT]]), (%[[LIN]] init: %[[NEG1S]]) across dimensions = [0, 2] : (tensor<2x3x2xf32>, tensor<2x3x2xi32>, tensor<f32>, tensor<i32>) -> (tensor<3xf32>, tensor<3xi32>)
// CHECK-NEXT:     reducer(%[[A:.+]]: tensor<f32>, %[[B:.+]]: tensor<f32>) (%[[IA:.+]]: tensor<i32>, %[[IB:.+]]: tensor<i32>)
// CHECK-NEXT:      %[[ANAN:.+]] = stablehlo.compare NE, %[[A]], %[[A]]
// CHECK-NEXT:      %[[BNAN:.+]] = stablehlo.compare NE, %[[B]], %[[B]]
// CHECK-NEXT:      %[[GT:.+]] = stablehlo.compare GT, %[[A]], %[[B]]
// CHECK-NEXT:      %[[NBNAN:.+]] = stablehlo.not %[[BNAN]]
// CHECK-NEXT:      %[[ONLYANAN:.+]] = stablehlo.and %[[ANAN]], %[[NBNAN]]
// CHECK-NEXT:      %[[BETTER:.+]] = stablehlo.or %[[GT]], %[[ONLYANAN]]
// CHECK-NEXT:      %[[EQ:.+]] = stablehlo.compare EQ, %[[A]], %[[B]]
// CHECK-NEXT:      %[[BOTHNAN:.+]] = stablehlo.and %[[ANAN]], %[[BNAN]]
// CHECK-NEXT:      %[[SAME:.+]] = stablehlo.or %[[EQ]], %[[BOTHNAN]]
// CHECK-NEXT:      %[[LATER:.+]] = stablehlo.compare GT, %[[IA]], %[[IB]]
// CHECK-NEXT:      %[[TIE:.+]] = stablehlo.and %[[SAME]], %[[LATER]]
// CHECK-NEXT:      %[[PICKA:.+]] = stablehlo.or %[[BETTER]], %[[TIE]]
// CHECK-NEXT:      %[[V:.+]] = stablehlo.select %[[PICKA]], %[[A]], %[[B]]
// CHECK-NEXT:      %[[I:.+]] = stablehlo.select %[[PICKA]], %[[IA]], %[[IB]]
// CHECK-NEXT:      stablehlo.return %[[V]], %[[I]]
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RIOTA2:.+]] = stablehlo.iota dim = 2 : tensor<2x3x2xi32>
// CHECK-NEXT:    %[[RIOTA0:.+]] = stablehlo.iota dim = 0 : tensor<2x3x2xi32>
// CHECK-NEXT:    %[[RSCALED:.+]] = stablehlo.multiply %[[RIOTA0]], %[[STRIDE]]
// CHECK-NEXT:    %[[RLIN:.+]] = stablehlo.add %[[RIOTA2]], %[[RSCALED]]
// CHECK-NEXT:    %[[WINBC:.+]] = stablehlo.broadcast_in_dim %[[RED]]#1, dims = [1] : (tensor<3xi32>) -> tensor<2x3x2xi32>
// CHECK-NEXT:    %[[ISWIN:.+]] = stablehlo.compare EQ, %[[RLIN]], %[[WINBC]]
// CHECK-NEXT:    %[[DRESBC:.+]] = stablehlo.broadcast_in_dim %[[DRES]], dims = [1] : (tensor<3xf32>) -> tensor<2x3x2xf32>
// CHECK-NEXT:    %[[DX:.+]] = stablehlo.select %[[ISWIN]], %[[DRESBC]], %[[ZERO]]
// CHECK-NEXT:    %[[FROMINIT:.+]] = stablehlo.compare EQ, %[[RED]]#1, %[[NEG1R]]
// CHECK-NEXT:    %[[DINITR:.+]] = stablehlo.select %[[FROMINIT]], %[[DRES]], %[[ZEROR]]
// CHECK-NEXT:    %[[DINIT:.+]] = stablehlo.reduce(%[[DINITR]] init: %[[ZEROS]]) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    return %[[DX]], %[[DINIT]]

// CHECK-LABEL: func.func private @diffeg(
// CHECK-SAME:      %[[X:.+]]: tensor<2x3xf32>, %[[INIT:.+]]: tensor<f32>, %[[DRES:.+]]: tensor<2xf32>)
// CHECK:         %[[LIN:.+]] = stablehlo.iota dim = 1 : tensor<2x3xi32>
// CHECK-NEXT:    %[[RED:.+]]:2 = stablehlo.reduce(%[[X]] init: %[[INIT]]), (%[[LIN]] init: %{{.+}}) across dimensions = [1] : (tensor<2x3xf32>, tensor<2x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<2xf32>, tensor<2xi32>)
// CHECK:           stablehlo.compare LT,
// CHECK:         %[[RLIN:.+]] = stablehlo.iota dim = 1 : tensor<2x3xi32>
// CHECK-NEXT:    %[[WINBC:.+]] = stablehlo.broadcast_in_dim %[[RED]]#1, dims = [0] : (tensor<2xi32>) -> tensor<2x3xi32>
// CHECK-NEXT:    %[[ISWIN:.+]] = stablehlo.compare EQ, %[[RLIN]], %[[WINBC]]
