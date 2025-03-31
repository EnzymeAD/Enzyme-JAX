// RUN: enzymexlamlir-opt  -allow-unregistered-dialect --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=65536})" %s | FileCheck %s

func.func @test_while_transpose_elimination(%arg0: tensor<2x3xf32>, %arg1: tensor<i64>) -> tensor<3x2xf32> {
  %cond = stablehlo.constant dense<true> : tensor<i1>
  
  %0:2 = "stablehlo.while"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<2x3xf32>, %arg3: tensor<i64>):
      stablehlo.return %cond : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<2x3xf32>, %arg3: tensor<i64>):
      // Transpose input inside while body
      %transposed = "unregistered.custom_transform"(%arg2) {} : (tensor<2x3xf32>) -> tensor<3x2xf32>
      %transposed_back = stablehlo.transpose %transposed, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
      
      // Do something with the iteration count
      %one = stablehlo.constant dense<1> : tensor<i64>
      %next_iter = stablehlo.add %arg3, %one : tensor<i64>
      
      // Yield the transposed value (which becomes a result of the while)
      stablehlo.return %transposed_back, %next_iter : tensor<2x3xf32>, tensor<i64>
  }) : (tensor<2x3xf32>, tensor<i64>) -> (tensor<2x3xf32>, tensor<i64>)
  
  // Transpose the while result back to the original shape (this should be eliminated by the pattern)
  %1 = stablehlo.transpose %0#0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  
  return %1 : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @test_while_transpose_elimination
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3xf32>, %[[ARG1:.*]]: tensor<i64>) -> tensor<3x2xf32>
// CHECK-NEXT: %[[COND:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT: %[[TRANSPOSE_BEFORE:.*]] = stablehlo.transpose %[[ARG0]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT: %[[WHILE_RESULT:.*]] = stablehlo.while(%[[ITER_ARG:.*]] = %[[TRANSPOSE_BEFORE]]) : tensor<3x2xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     stablehlo.return %[[COND]] : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %[[BODY_TRANS:.*]] = stablehlo.transpose %[[ITER_ARG]], dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %[[CUSTOM:.*]] = "unregistered.custom_transform"(%[[BODY_TRANS]]) : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:     stablehlo.return %[[CUSTOM]] : tensor<3x2xf32>
// CHECK:        }
// CHECK-NEXT: return %[[WHILE_RESULT]] : tensor<3x2xf32>