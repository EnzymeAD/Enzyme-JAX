// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform --inline --enzyme-hlo-opt %s | FileCheck %s

// The scale factor is a chain of 40 adds of a value to itself, computed in the
// loop body from a loop-invariant operand. Deciding that it is invariant used
// to walk each add once per path reaching it, 2^40 times in all, and to
// collect the hoistable chain with the same multiplicity; the test only
// finishes because every op is now decided once. The chain is then hoisted and
// the multiply batched out of the loop.

module {
  func.func @main(%arg0: tensor<8x4xf32>, %arg1: tensor<4xf32>) -> tensor<8x4xf32> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<8> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst) : tensor<i64>, tensor<8x4xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %d0 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
      %d1 = stablehlo.add %d0, %d0 : tensor<4xf32>
      %d2 = stablehlo.add %d1, %d1 : tensor<4xf32>
      %d3 = stablehlo.add %d2, %d2 : tensor<4xf32>
      %d4 = stablehlo.add %d3, %d3 : tensor<4xf32>
      %d5 = stablehlo.add %d4, %d4 : tensor<4xf32>
      %d6 = stablehlo.add %d5, %d5 : tensor<4xf32>
      %d7 = stablehlo.add %d6, %d6 : tensor<4xf32>
      %d8 = stablehlo.add %d7, %d7 : tensor<4xf32>
      %d9 = stablehlo.add %d8, %d8 : tensor<4xf32>
      %d10 = stablehlo.add %d9, %d9 : tensor<4xf32>
      %d11 = stablehlo.add %d10, %d10 : tensor<4xf32>
      %d12 = stablehlo.add %d11, %d11 : tensor<4xf32>
      %d13 = stablehlo.add %d12, %d12 : tensor<4xf32>
      %d14 = stablehlo.add %d13, %d13 : tensor<4xf32>
      %d15 = stablehlo.add %d14, %d14 : tensor<4xf32>
      %d16 = stablehlo.add %d15, %d15 : tensor<4xf32>
      %d17 = stablehlo.add %d16, %d16 : tensor<4xf32>
      %d18 = stablehlo.add %d17, %d17 : tensor<4xf32>
      %d19 = stablehlo.add %d18, %d18 : tensor<4xf32>
      %d20 = stablehlo.add %d19, %d19 : tensor<4xf32>
      %d21 = stablehlo.add %d20, %d20 : tensor<4xf32>
      %d22 = stablehlo.add %d21, %d21 : tensor<4xf32>
      %d23 = stablehlo.add %d22, %d22 : tensor<4xf32>
      %d24 = stablehlo.add %d23, %d23 : tensor<4xf32>
      %d25 = stablehlo.add %d24, %d24 : tensor<4xf32>
      %d26 = stablehlo.add %d25, %d25 : tensor<4xf32>
      %d27 = stablehlo.add %d26, %d26 : tensor<4xf32>
      %d28 = stablehlo.add %d27, %d27 : tensor<4xf32>
      %d29 = stablehlo.add %d28, %d28 : tensor<4xf32>
      %d30 = stablehlo.add %d29, %d29 : tensor<4xf32>
      %d31 = stablehlo.add %d30, %d30 : tensor<4xf32>
      %d32 = stablehlo.add %d31, %d31 : tensor<4xf32>
      %d33 = stablehlo.add %d32, %d32 : tensor<4xf32>
      %d34 = stablehlo.add %d33, %d33 : tensor<4xf32>
      %d35 = stablehlo.add %d34, %d34 : tensor<4xf32>
      %d36 = stablehlo.add %d35, %d35 : tensor<4xf32>
      %d37 = stablehlo.add %d36, %d36 : tensor<4xf32>
      %d38 = stablehlo.add %d37, %d37 : tensor<4xf32>
      %d39 = stablehlo.add %d38, %d38 : tensor<4xf32>
      %1 = stablehlo.dynamic_slice %arg0, %iterArg, %c, sizes = [1, 4] : (tensor<8x4xf32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
      %2 = stablehlo.reshape %1 : (tensor<1x4xf32>) -> tensor<4xf32>
      %3 = stablehlo.multiply %2, %d39 : tensor<4xf32>
      %4 = stablehlo.reshape %3 : (tensor<4xf32>) -> tensor<1x4xf32>
      %5 = stablehlo.dynamic_update_slice %iterArg_2, %4, %iterArg, %c : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
      %6 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      stablehlo.return %6, %5 : tensor<i64>, tensor<8x4xf32>
    }
    return %0#1 : tensor<8x4xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<8x4xf32>, %[[ARG1:.+]]: tensor<4xf32>)
// CHECK:         %[[D0:.+]] = stablehlo.add %[[ARG1]], %[[ARG1]] : tensor<4xf32>
// CHECK:         %[[D1:.+]] = stablehlo.add %[[D0]], %[[D0]] : tensor<4xf32>
// CHECK:         %[[SCALE:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [1] : (tensor<4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT:    %[[PROD:.+]] = stablehlo.multiply %[[ARG0]], %[[SCALE]] : tensor<8x4xf32>
// CHECK-NEXT:    %[[WHILE:.+]]:2 = stablehlo.while
// CHECK:         } do {
// CHECK-NEXT:      %[[ROW:.+]] = stablehlo.dynamic_slice %[[PROD]], %iterArg, %c, sizes = [1, 4]
// CHECK-NEXT:      %[[DUS:.+]] = stablehlo.dynamic_update_slice %iterArg_2, %[[ROW]], %iterArg, %c
// CHECK-NOT:       stablehlo.multiply
// CHECK:         return %[[WHILE]]#1
