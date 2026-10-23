// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --drop-unsupported-attributes | stablehlo-translate - --interpret
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s

// When several elements tie with the result of a reduce maximum/minimum, only
// the last one in row-major order over the reduced dimensions gets the
// adjoint. The init value gets it only when no input element matches.

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
  func.return
}

// CHECK-LABEL: func.func private @diffef(
// CHECK-SAME:      %[[X:.+]]: tensor<2x3x2xf32>, %[[INIT:.+]]: tensor<f32>, %[[DRES:.+]]: tensor<3xf32>)
// CHECK-NEXT:    %[[NEG1R:.+]] = stablehlo.constant dense<-1> : tensor<3xi64>
// CHECK-NEXT:    %[[NEG1S:.+]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %[[NEG1:.+]] = stablehlo.constant dense<-1> : tensor<2x3x2xi64>
// CHECK-NEXT:    %[[STRIDE:.+]] = stablehlo.constant dense<2> : tensor<2x3x2xi64>
// CHECK-NEXT:    %[[ZEROR:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x3x2xf32>
// CHECK-NEXT:    %[[ZEROS:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[RES:.+]] = stablehlo.reduce(%[[X]] init: %[[INIT]]) applies stablehlo.maximum across dimensions = [0, 2]
// CHECK-NEXT:    %[[IOTA2:.+]] = stablehlo.iota dim = 2 : tensor<2x3x2xi64>
// CHECK-NEXT:    %[[IOTA0:.+]] = stablehlo.iota dim = 0 : tensor<2x3x2xi64>
// CHECK-NEXT:    %[[SCALED:.+]] = stablehlo.multiply %[[IOTA0]], %[[STRIDE]]
// CHECK-NEXT:    %[[LIN:.+]] = stablehlo.add %[[IOTA2]], %[[SCALED]]
// CHECK-NEXT:    %[[RESBC:.+]] = stablehlo.broadcast_in_dim %[[RES]], dims = [1] : (tensor<3xf32>) -> tensor<2x3x2xf32>
// CHECK-NEXT:    %[[MATCH:.+]] = stablehlo.compare EQ, %[[RESBC]], %[[X]]
// CHECK-NEXT:    %[[MASKED:.+]] = stablehlo.select %[[MATCH]], %[[LIN]], %[[NEG1]]
// CHECK-NEXT:    %[[LAST:.+]] = stablehlo.reduce(%[[MASKED]] init: %[[NEG1S]]) applies stablehlo.maximum across dimensions = [0, 2] : (tensor<2x3x2xi64>, tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[LASTBC:.+]] = stablehlo.broadcast_in_dim %[[LAST]], dims = [1] : (tensor<3xi64>) -> tensor<2x3x2xi64>
// CHECK-NEXT:    %[[ISLAST:.+]] = stablehlo.compare EQ, %[[LIN]], %[[LASTBC]]
// CHECK-NEXT:    %[[DRESBC:.+]] = stablehlo.broadcast_in_dim %[[DRES]], dims = [1] : (tensor<3xf32>) -> tensor<2x3x2xf32>
// CHECK-NEXT:    %[[DX:.+]] = stablehlo.select %[[ISLAST]], %[[DRESBC]], %[[ZERO]]
// CHECK-NEXT:    %[[FROMINIT:.+]] = stablehlo.compare EQ, %[[LAST]], %[[NEG1R]]
// CHECK-NEXT:    %[[DINITR:.+]] = stablehlo.select %[[FROMINIT]], %[[DRES]], %[[ZEROR]]
// CHECK-NEXT:    %[[DINIT:.+]] = stablehlo.reduce(%[[DINITR]] init: %[[ZEROS]]) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    return %[[DX]], %[[DINIT]]

// CHECK-LABEL: func.func private @diffeg(
// CHECK-SAME:      %[[X:.+]]: tensor<2x3xf32>, %[[INIT:.+]]: tensor<f32>, %[[DRES:.+]]: tensor<2xf32>)
// CHECK-NEXT:    %[[NEG1R:.+]] = stablehlo.constant dense<-1> : tensor<2xi64>
// CHECK-NEXT:    %[[NEG1S:.+]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %[[NEG1:.+]] = stablehlo.constant dense<-1> : tensor<2x3xi64>
// CHECK-NEXT:    %[[ZEROR:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf32>
// CHECK-NEXT:    %[[ZEROS:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[RES:.+]] = stablehlo.reduce(%[[X]] init: %[[INIT]]) applies stablehlo.minimum across dimensions = [1]
// CHECK-NEXT:    %[[LIN:.+]] = stablehlo.iota dim = 1 : tensor<2x3xi64>
// CHECK-NEXT:    %[[RESBC:.+]] = stablehlo.broadcast_in_dim %[[RES]], dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %[[MATCH:.+]] = stablehlo.compare EQ, %[[RESBC]], %[[X]]
// CHECK-NEXT:    %[[MASKED:.+]] = stablehlo.select %[[MATCH]], %[[LIN]], %[[NEG1]]
// CHECK-NEXT:    %[[LAST:.+]] = stablehlo.reduce(%[[MASKED]] init: %[[NEG1S]]) applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xi64>, tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[LASTBC:.+]] = stablehlo.broadcast_in_dim %[[LAST]], dims = [0] : (tensor<2xi64>) -> tensor<2x3xi64>
// CHECK-NEXT:    %[[ISLAST:.+]] = stablehlo.compare EQ, %[[LIN]], %[[LASTBC]]
// CHECK-NEXT:    %[[DRESBC:.+]] = stablehlo.broadcast_in_dim %[[DRES]], dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %[[DX:.+]] = stablehlo.select %[[ISLAST]], %[[DRESBC]], %[[ZERO]]
// CHECK-NEXT:    %[[FROMINIT:.+]] = stablehlo.compare EQ, %[[LAST]], %[[NEG1R]]
// CHECK-NEXT:    %[[DINITR:.+]] = stablehlo.select %[[FROMINIT]], %[[DRES]], %[[ZEROR]]
// CHECK-NEXT:    %[[DINIT:.+]] = stablehlo.reduce(%[[DINITR]] init: %[[ZEROS]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    return %[[DX]], %[[DINIT]]
