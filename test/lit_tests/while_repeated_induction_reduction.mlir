// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_repeated_induction_reduction;slice_concat;slice_if;concat_const_prop;noop_slice;slice_licm(1);cse_slice;if_to_select" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect | FileCheck %s

func.func @while_loop_example(%arg0: tensor<1x2034x2032xf64>, %arg1: tensor<i32>, %lim : tensor<i32>) -> tensor<1x2034x2032xf64> {
  %cst_0 = stablehlo.constant dense<0> : tensor<i32>

  // Increment counter
  %one = stablehlo.constant dense<1> : tensor<i32>

  // Initialize loop state
  %init = stablehlo.constant dense<0.0> : tensor<1x2034x2032xf64>
  
  // Start the while loop with counter and tensor - using the older format for StableHLO
  %results:2 = "stablehlo.while"(%cst_0, %arg0) ({
    ^bb0(%counter: tensor<i32>, %tensor: tensor<1x2034x2032xf64>):
      // CONDITION: counter < 10
      %pred = stablehlo.compare LT, %counter, %lim : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %pred : tensor<i1>
  }, {
    ^bb0(%iter_counter: tensor<i32>, %iter_tensor: tensor<1x2034x2032xf64>):
      %next_counter = stablehlo.add %iter_counter, %one : tensor<i32>
      
      %next_tensor = "test.create"(%iter_tensor) : (tensor<1x2034x2032xf64>) -> (tensor<1x2032x2032xf64>)

      // Slice operations as in your example
      %slice_top = stablehlo.slice %next_tensor [0:1, 0:1, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
      %slice_bottom = stablehlo.slice %next_tensor [0:1, 2031:2032, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
      
      // Concatenate operation
      %concat = stablehlo.concatenate %slice_top, %next_tensor, %slice_bottom, dim = 1 : (tensor<1x1x2032xf64>, tensor<1x2032x2032xf64>, tensor<1x1x2032xf64>) -> tensor<1x2034x2032xf64>
      
      // Return for next iteration
      stablehlo.return %next_counter, %concat : tensor<i32>, tensor<1x2034x2032xf64>
  }) : (tensor<i32>, tensor<1x2034x2032xf64>) -> (tensor<i32>, tensor<1x2034x2032xf64>)
  
  // Return the final result
  return %results#1 : tensor<1x2034x2032xf64>
}


func.func @while_loop_example2(%arg0: tensor<1x2034x2032xf64>, %arg1: tensor<i32>, %lim : tensor<i32>) -> tensor<1x2034x2032xf64> {
  %cst_0 = stablehlo.constant dense<0> : tensor<i32>

  // Increment counter
  %one = stablehlo.constant dense<1> : tensor<i32>

  // Initialize loop state
  %init = stablehlo.constant dense<0.0> : tensor<1x2034x2032xf64>
  
  // Start the while loop with counter and tensor - using the older format for StableHLO
  %results:2 = "stablehlo.while"(%cst_0, %arg0) ({
    ^bb0(%counter: tensor<i32>, %tensor: tensor<1x2034x2032xf64>):
      // CONDITION: counter < 10
      %pred = stablehlo.compare LT, %counter, %lim : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %pred : tensor<i1>
  }, {
    ^bb0(%iter_counter: tensor<i32>, %iter_tensor: tensor<1x2034x2032xf64>):
      %next_counter = stablehlo.add %iter_counter, %one : tensor<i32>
      
      %mid_tensor =  stablehlo.slice %iter_tensor [0:1, 1:2033, 0:2032] : (tensor<1x2034x2032xf64>) -> tensor<1x2032x2032xf64>

      %next_tensor = "test.create"(%mid_tensor) : (tensor<1x2032x2032xf64>) -> (tensor<1x2032x2032xf64>)

      // Slice operations as in your example
      %slice_top = stablehlo.slice %next_tensor [0:1, 0:1, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
      %slice_bottom = stablehlo.slice %next_tensor [0:1, 2031:2032, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
      
      // Concatenate operation
      %concat = stablehlo.concatenate %slice_top, %next_tensor, %slice_bottom, dim = 1 : (tensor<1x1x2032xf64>, tensor<1x2032x2032xf64>, tensor<1x1x2032xf64>) -> tensor<1x2034x2032xf64>
      
      // Return for next iteration
      stablehlo.return %next_counter, %concat : tensor<i32>, tensor<1x2034x2032xf64>
  }) : (tensor<i32>, tensor<1x2034x2032xf64>) -> (tensor<i32>, tensor<1x2034x2032xf64>)
  
  // Return the final result
  return %results#1 : tensor<1x2034x2032xf64>
}

// CHECK:  func.func @while_loop_example(%arg0: tensor<1x2034x2032xf64>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<1x2034x2032xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 1:2033, 0:2032] : (tensor<1x2034x2032xf64>) -> tensor<1x2032x2032xf64>
// CHECK-NEXT:    %1:2 = stablehlo.while(%iterArg = %c, %iterArg_1 = %0) : tensor<i32>, tensor<1x2032x2032xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %4 = stablehlo.compare  LT, %iterArg, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.compare  EQ, %iterArg, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:      %5 = "stablehlo.if"(%4) ({
// CHECK-NEXT:        stablehlo.return %arg0 : tensor<1x2034x2032xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %8 = stablehlo.slice %iterArg_1 [0:1, 0:1, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
// CHECK-NEXT:        %9 = stablehlo.slice %iterArg_1 [0:1, 2031:2032, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
// CHECK-NEXT:        %10 = stablehlo.concatenate %8, %iterArg_1, %9, dim = 1 : (tensor<1x1x2032xf64>, tensor<1x2032x2032xf64>, tensor<1x1x2032xf64>) -> tensor<1x2034x2032xf64>
// CHECK-NEXT:        stablehlo.return %10 : tensor<1x2034x2032xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<1x2034x2032xf64>
// CHECK-NEXT:      %6 = stablehlo.add %iterArg, %c_0 : tensor<i32>
// CHECK-NEXT:      %7 = "test.create"(%5) : (tensor<1x2034x2032xf64>) -> tensor<1x2032x2032xf64>
// CHECK-NEXT:      stablehlo.return %6, %7 : tensor<i32>, tensor<1x2032x2032xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = stablehlo.compare  EQ, %c, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:    %3 = "stablehlo.if"(%2) ({
// CHECK-NEXT:      stablehlo.return %arg0 : tensor<1x2034x2032xf64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %4 = stablehlo.slice %1#1 [0:1, 0:1, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
// CHECK-NEXT:      %5 = stablehlo.slice %1#1 [0:1, 2031:2032, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
// CHECK-NEXT:      %6 = stablehlo.concatenate %4, %1#1, %5, dim = 1 : (tensor<1x1x2032xf64>, tensor<1x2032x2032xf64>, tensor<1x1x2032xf64>) -> tensor<1x2034x2032xf64>
// CHECK-NEXT:      stablehlo.return %6 : tensor<1x2034x2032xf64>
// CHECK-NEXT:    }) : (tensor<i1>) -> tensor<1x2034x2032xf64>
// CHECK-NEXT:    return %3 : tensor<1x2034x2032xf64>
// CHECK-NEXT:  }


// CHECK:  func.func @while_loop_example2(%arg0: tensor<1x2034x2032xf64>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<1x2034x2032xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 1:2033, 0:2032] : (tensor<1x2034x2032xf64>) -> tensor<1x2032x2032xf64>
// CHECK-NEXT:    %1:2 = stablehlo.while(%iterArg = %c, %iterArg_1 = %0) : tensor<i32>, tensor<1x2032x2032xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %4 = stablehlo.compare  LT, %iterArg, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.compare  EQ, %iterArg, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c_0 : tensor<i32>
// CHECK-NEXT:      %6 = stablehlo.select %4, %0, %iterArg_1 : tensor<i1>, tensor<1x2032x2032xf64>
// CHECK-NEXT:      %7 = "test.create"(%6) : (tensor<1x2032x2032xf64>) -> tensor<1x2032x2032xf64>
// CHECK-NEXT:      stablehlo.return %5, %7 : tensor<i32>, tensor<1x2032x2032xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = stablehlo.compare  EQ, %c, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:    %3 = "stablehlo.if"(%2) ({
// CHECK-NEXT:      stablehlo.return %arg0 : tensor<1x2034x2032xf64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %4 = stablehlo.slice %1#1 [0:1, 0:1, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
// CHECK-NEXT:      %5 = stablehlo.slice %1#1 [0:1, 2031:2032, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x1x2032xf64>
// CHECK-NEXT:      %6 = stablehlo.concatenate %4, %1#1, %5, dim = 1 : (tensor<1x1x2032xf64>, tensor<1x2032x2032xf64>, tensor<1x1x2032xf64>) -> tensor<1x2034x2032xf64>
// CHECK-NEXT:      stablehlo.return %6 : tensor<1x2034x2032xf64>
// CHECK-NEXT:    }) : (tensor<i1>) -> tensor<1x2034x2032xf64>
// CHECK-NEXT:    return %3 : tensor<1x2034x2032xf64>
// CHECK-NEXT:  }