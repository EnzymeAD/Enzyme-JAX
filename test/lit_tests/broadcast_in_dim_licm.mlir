// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=broadcastindim_licm(0)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// An expanding broadcast stays in the loop body; a non-expanding one is hoisted.
module {
  func.func @main(%arg0: tensor<161xf64>, %arg1: tensor<4x161xf64>, %arg2: tensor<161x4xf64>) -> tensor<4x161xf64> {
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_10 = stablehlo.constant dense<10> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_0 = %arg1) : tensor<i64>, tensor<4x161xf64>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<161xf64>) -> tensor<4x161xf64>
      %2 = stablehlo.broadcast_in_dim %arg2, dims = [1, 0] : (tensor<161x4xf64>) -> tensor<4x161xf64>
      %3 = stablehlo.multiply %iterArg_0, %1 : tensor<4x161xf64>
      %4 = stablehlo.add %3, %2 : tensor<4x161xf64>
      %5 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      stablehlo.return %5, %4 : tensor<i64>, tensor<4x161xf64>
    }
    return %0#1 : tensor<4x161xf64>
  }
}

// CHECK-LABEL: func.func @main
// CHECK:      %[[SAME:.+]] = stablehlo.broadcast_in_dim %arg2, dims = [1, 0] : (tensor<161x4xf64>) -> tensor<4x161xf64>
// CHECK-NEXT: stablehlo.while
// CHECK:      } do {
// CHECK-NEXT:   %[[GROW:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<161xf64>) -> tensor<4x161xf64>
// CHECK-NEXT:   stablehlo.multiply %{{.+}}, %[[GROW]]

// An expanding broadcast whose reduce user becomes invariant is hoisted, so
// the reduce can follow it out of the loop.
module {
  func.func @reduce_user(%arg0: tensor<161xf64>, %arg1: tensor<161xf64>) -> tensor<161xf64> {
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_10 = stablehlo.constant dense<10> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_0 = %arg1) : tensor<i64>, tensor<161xf64>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<161xf64>) -> tensor<4x161xf64>
      %2 = stablehlo.reduce(%1 init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<4x161xf64>, tensor<f64>) -> tensor<161xf64>
      %3 = stablehlo.add %iterArg_0, %2 : tensor<161xf64>
      %4 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      stablehlo.return %4, %3 : tensor<i64>, tensor<161xf64>
    }
    return %0#1 : tensor<161xf64>
  }
}

// CHECK-LABEL: func.func @reduce_user
// CHECK:      stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<161xf64>) -> tensor<4x161xf64>
// CHECK:      stablehlo.while
