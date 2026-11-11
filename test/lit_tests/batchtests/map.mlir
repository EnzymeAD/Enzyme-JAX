// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func private @map_sqrt(%arg0: tensor<3xf64>) -> tensor<3xf64> {
  %0 = "stablehlo.map"(%arg0) <{dimensions = array<i64: 0>}> ({
  ^bb0(%arg1: tensor<f64>):
    %1 = stablehlo.sqrt %arg1 : tensor<f64>
    stablehlo.return %1 : tensor<f64>
  }) : (tensor<3xf64>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
}

func.func private @map_binary_const(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
  %scale = stablehlo.constant dense<2.0> : tensor<f64>
  %0 = "stablehlo.map"(%arg0, %arg1) <{dimensions = array<i64: 0, 1>}> ({
  ^bb0(%lhs: tensor<f64>, %rhs: tensor<f64>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f64>
    %prod = stablehlo.multiply %sum, %scale : tensor<f64>
    stablehlo.return %prod : tensor<f64>
  }) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}

func.func @main() {
  %in = stablehlo.constant dense<[[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]> : tensor<2x3xf64>
  %res0 = enzyme.batch @map_sqrt(%in) <{batch_shape = array<i64: 2>}> : (tensor<2x3xf64>) -> tensor<2x3xf64>
  %exp0 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  "check.expect_almost_eq"(%res0, %exp0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()

  %a = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf64>
  %b = stablehlo.constant dense<[[[10.0, 20.0], [30.0, 40.0]], [[1.0, 2.0], [3.0, 4.0]]]> : tensor<2x2x2xf64>
  %res1 = enzyme.batch @map_binary_const(%a, %b) <{batch_shape = array<i64: 2>}> : (tensor<2x2x2xf64>, tensor<2x2x2xf64>) -> tensor<2x2x2xf64>
  %exp1 = stablehlo.constant dense<[[[22.0, 44.0], [66.0, 88.0]], [[12.0, 16.0], [20.0, 24.0]]]> : tensor<2x2x2xf64>
  "check.expect_almost_eq"(%res1, %exp1) : (tensor<2x2x2xf64>, tensor<2x2x2xf64>) -> ()
  return
}

// CHECK-LABEL: func.func private @batched_map_sqrt(
// CHECK-SAME: %arg0: tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK: %[[MAP:.+]] = "stablehlo.map"(%arg0) <{dimensions = array<i64: 0, 1>}> ({
// CHECK-NEXT: ^bb0(%arg1: tensor<f64>):
// CHECK-NEXT:   %[[SQRT:.+]] = stablehlo.sqrt %arg1 : tensor<f64>
// CHECK-NEXT:   stablehlo.return %[[SQRT]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK: return %[[MAP]] : tensor<2x3xf64>
