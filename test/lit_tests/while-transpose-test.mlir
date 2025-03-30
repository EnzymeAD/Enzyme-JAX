// RUN: enzymexlamlir-opt  -allow-unregistered-dialect --enzyme-hlo-opt %s | FileCheck %s

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
  
  // Use the transposed result
  %2 = stablehlo.add %1, %1 : tensor<3x2xf32>
  
  return %2 : tensor<3x2xf32>
} 