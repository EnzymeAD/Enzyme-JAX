// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_pad_induction_reduction" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect | FileCheck %s

func.func @while_loop_example(%arg0: tensor<3x2034x2034xf64>, %arg1: tensor<i32>, %lim : tensor<i32>) -> tensor<3x2034x2034xf64> {
  %cst_0 = stablehlo.constant dense<0> : tensor<i32>

  // Increment counter
  %one = stablehlo.constant dense<1> : tensor<i32>

  // Initialize loop state
  %init = stablehlo.constant dense<0.0> : tensor<3x2034x2034xf64>
  
  // Start the while loop with counter and tensor - using the older format for StableHLO
  %results:2 = "stablehlo.while"(%cst_0, %arg0) ({
    ^bb0(%counter: tensor<i32>, %tensor: tensor<3x2034x2034xf64>):
      // CONDITION: counter < 10
      %pred = stablehlo.compare LT, %counter, %lim : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %pred : tensor<i1>
  }, {
    ^bb0(%iter_counter: tensor<i32>, %iter_tensor: tensor<3x2034x2034xf64>):
      %next_counter = stablehlo.add %iter_counter, %one : tensor<i32>
      
      %next_tensor = "test.create"(%iter_tensor) : (tensor<3x2034x2034xf64>) -> (tensor<1x2032x2032xf64>)

      // pad the tensor
      %padding_value = stablehlo.constant dense<0.0> : tensor<f64>
      %padded_tensor = stablehlo.pad %next_tensor, %padding_value, 
        low = [1, 1, 1], 
        high = [1, 1, 1], 
        interior = [0, 0, 0] 
        : (tensor<1x2032x2032xf64>, tensor<f64>) -> tensor<3x2034x2034xf64>
      // Return for next iteration
      stablehlo.return %next_counter, %padded_tensor : tensor<i32>, tensor<3x2034x2034xf64>
  }) : (tensor<i32>, tensor<3x2034x2034xf64>) -> (tensor<i32>, tensor<3x2034x2034xf64>)
  
  // Return the final result
  return %results#1 : tensor<3x2034x2034xf64>
}

// CHECK-LABEL: func.func @while_loop_example
// CHECK: %[[ARG0:[^:]+]]: tensor<3x2034x2034xf64>, %[[ARG1:[^:]+]]: tensor<i32>, %[[ARG2:[^:]+]]: tensor<i32>

// CHECK: %[[PADDING_VALUE:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK: %[[ONE:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK: %[[EMPTY_TENSOR:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x2032x2032xf64>

// CHECK: %[[WHILE_RESULT:.*]]:2 = stablehlo.while(%[[ITER_COUNTER:.*]] = %[[ZERO]], %[[ITER_TENSOR:.*]] = %[[EMPTY_TENSOR]])
// CHECK-SAME: tensor<i32>, tensor<1x2032x2032xf64>

// CHECK: cond {
// CHECK: %[[COND:.*]] = stablehlo.compare LT, %[[ITER_COUNTER]], %[[ARG2]]
// CHECK: stablehlo.return %[[COND]]

// CHECK: do {
// CHECK: %[[IS_FIRST:.*]] = stablehlo.compare EQ, %[[ITER_COUNTER]], %[[ZERO]]
// CHECK: %[[IF_RESULT:.*]] = "stablehlo.if"(%[[IS_FIRST]]) ({
// CHECK: stablehlo.return %[[ARG0]]
// CHECK: }, {
// CHECK: %[[PADDED:.*]] = stablehlo.pad %[[ITER_TENSOR]], %[[PADDING_VALUE]]
// CHECK-SAME: low = [1, 1, 1], high = [1, 1, 1], interior = [0, 0, 0]
// CHECK: stablehlo.return %[[PADDED]]
// CHECK: })

// CHECK: %[[NEXT_COUNTER:.*]] = stablehlo.add %[[ITER_COUNTER]], %[[ONE]]
// CHECK: %[[NEXT_TENSOR:.*]] = "test.create"(%[[IF_RESULT]])
// CHECK: stablehlo.return %[[NEXT_COUNTER]], %[[NEXT_TENSOR]]

// CHECK: %[[FINAL_COND:.*]] = stablehlo.compare LT, %[[ZERO]], %[[ARG2]]
// CHECK: %[[FINAL_RESULT:.*]] = "stablehlo.if"(%[[FINAL_COND]]) ({
// CHECK: %[[FINAL_PADDED:.*]] = stablehlo.pad %[[WHILE_RESULT]]#1, %[[PADDING_VALUE]]
// CHECK-SAME: low = [1, 1, 1], high = [1, 1, 1], interior = [0, 0, 0]
// CHECK: stablehlo.return %[[FINAL_PADDED]]
// CHECK: }, {
// CHECK: stablehlo.return %[[ARG0]]
// CHECK: })

// CHECK: return %[[FINAL_RESULT]]
