// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @while_loop_example(%arg0: tensor<1x2034x2032xf64>, %arg1: tensor<i32>) -> tensor<1x2034x2032xf64> {
  %cst_0 = stablehlo.constant dense<0> : tensor<i32>
  %cst_10 = stablehlo.constant dense<10> : tensor<i32>
  
  // Initialize loop state
  %init = stablehlo.constant dense<0.0> : tensor<1x2034x2032xf64>
  
  // Start the while loop with counter and tensor - using the older format for StableHLO
  %results:2 = "stablehlo.while"(%cst_0, %arg0) ({
    ^bb0(%counter: tensor<i32>, %tensor: tensor<1x2034x2032xf64>):
      // CONDITION: counter < 10
      %pred = stablehlo.compare LT, %counter, %cst_10 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %pred : tensor<i1>
  }, {
    ^bb0(%iter_counter: tensor<i32>, %iter_tensor: tensor<1x2034x2032xf64>):
      // Increment counter
      %one = stablehlo.constant dense<1> : tensor<i32>
      %next_counter = stablehlo.add %iter_counter, %one : tensor<i32>
      
      //Slice the tensor to get 1x2032x2032 
      %slice_tensor = stablehlo.slice %iter_tensor [0:1, 0:2032, 0:2032] : (tensor<1x2034x2032xf64>) -> tensor<1x2032x2032xf64>

      // Slice operations as in your example
      %slice_top = stablehlo.slice %slice_tensor [0:1, 0:1, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
      %slice_bottom = stablehlo.slice %slice_tensor [0:1, 2031:2032, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
      
      // Concatenate operation
      %concat = stablehlo.concatenate %slice_top, %slice_tensor, %slice_bottom, dim = 1 : (tensor<1x1x2032xf64>, tensor<1x2032x2032xf64>, tensor<1x1x2032xf64>) -> tensor<1x2034x2032xf64>
      
      // Return for next iteration
      stablehlo.return %next_counter, %concat : tensor<i32>, tensor<1x2034x2032xf64>
  }) : (tensor<i32>, tensor<1x2034x2032xf64>) -> (tensor<i32>, tensor<1x2034x2032xf64>)
  
  // Return the final result
  return %results#1 : tensor<1x2034x2032xf64>
}