// RUN: env-enzyme-opt %s --enzyme-hlo-opt -split-input-file | FileCheck %s

func.func @test_while_induction_replacement(%other_value: tensor<i32>) -> tensor<i32> {
  %init_counter = stablehlo.constant dense<0> : tensor<i32>
  %init_sum = stablehlo.constant dense<0> : tensor<i32>
  %limit = stablehlo.constant dense<10> : tensor<i32>
  %step = stablehlo.constant dense<1> : tensor<i32>
  %sum_step = stablehlo.constant dense<2> : tensor<i32>

  %results:2 = stablehlo.while(%counter = %init_counter, %sum = %init_sum) : tensor<i32>, tensor<i32>
      cond {
        %cmp = stablehlo.compare LT, %counter, %limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %cmp : tensor<i1>
      } do {
        %new_counter = stablehlo.add %counter, %step : tensor<i32>
        
        // This is the induction variable we're optimizing:
        %new_sum = stablehlo.add %sum, %sum_step : tensor<i32>
        
        stablehlo.return %new_counter, %new_sum : tensor<i32>, tensor<i32>
      }

  // Use sum result later
  %final_result = stablehlo.add %results#1, %other_value : tensor<i32>
  
  return %final_result : tensor<i32>
}

// CHECK-LABEL: func.func @test_while_induction_replacement(
// CHECK-SAME:    %[[OTHER_VALUE:.*]]: tensor<i32>) -> tensor<i32> {
// CHECK:       %[[INIT_COUNTER:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK:       %[[INIT_SUM:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK:       %[[LIMIT:.*]] = stablehlo.constant dense<10> : tensor<i32>
// CHECK:       %[[STEP:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:       %[[SUM_STEP:.*]] = stablehlo.constant dense<2> : tensor<i32>

// CHECK:       %[[RESULTS:.*]]:2 = stablehlo.while(%[[COUNTER:.*]] = %[[INIT_COUNTER]]
// CHECK-NOT:                                      %[[SUM:.*]] = %[[INIT_SUM]])
// CHECK:         cond {
// CHECK:           %[[CMP:.*]] = stablehlo.compare LT, %[[COUNTER]], %[[LIMIT]]
// CHECK:           stablehlo.return %[[CMP]] : tensor<i1>
// CHECK:         } do {
// CHECK:           // Inside the loop, expect the calculation formula instead of iterative addition
// CHECK:           %[[OFFSET:.*]] = stablehlo.subtract %[[COUNTER]], %[[INIT_COUNTER]] : tensor<i32>
// CHECK:           %[[SCALED:.*]] = stablehlo.multiply %[[OFFSET]], %[[SUM_STEP]] : tensor<i32>
// CHECK:           %[[CALCULATED_SUM:.*]] = stablehlo.add %[[INIT_SUM]], %[[SCALED]] : tensor<i32>
// CHECK:           %[[NEW_COUNTER:.*]] = stablehlo.add %[[COUNTER]], %[[STEP]] : tensor<i32>
// CHECK:           stablehlo.return %[[NEW_COUNTER]], %[[CALCULATED_SUM]] : tensor<i32>, tensor<i32>
// CHECK:         }

// CHECK:       // Outside the loop, expect direct calculation of final sum
// CHECK:       %[[TOTAL_ITERS:.*]] = stablehlo.subtract %[[LIMIT]], %[[INIT_COUNTER]] : tensor<i32>
// CHECK:       %[[TOTAL_INCR:.*]] = stablehlo.multiply %[[TOTAL_ITERS]], %[[SUM_STEP]] : tensor<i32>
// CHECK:       %[[FINAL_SUM:.*]] = stablehlo.add %[[INIT_SUM]], %[[TOTAL_INCR]] : tensor<i32>

// CHECK:       %[[RESULT:.*]] = stablehlo.add %[[FINAL_SUM]], %[[OTHER_VALUE]] : tensor<i32>
// CHECK:       return %[[RESULT]] : tensor<i32>