// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt -allow-unregistered-dialect | FileCheck %s

func.func @test_while_induction_2(%other_value: tensor<i32>, %dynamic_limit: tensor<i32>) -> tensor<i32> {
  %init_counter = stablehlo.constant dense<7> : tensor<i32>
  %init_sum = stablehlo.constant dense<3> : tensor<i32>
  %step = stablehlo.constant dense<2> : tensor<i32>
  %sum_step = stablehlo.constant dense<5> : tensor<i32>
  
  // Create an initial value using an unregistered dialect
  %custom_init = "test.unknown_state"() : () -> tensor<i32>
  
  // Now using three iteration arguments (counter, sum, custom)
  %results:3 = stablehlo.while(%counter = %init_counter, %sum = %init_sum, %custom_val = %custom_init) : tensor<i32>, tensor<i32>, tensor<i32>
      cond {
        %cmp = stablehlo.compare LT, %counter, %dynamic_limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %cmp : tensor<i1>
      } do {
        %new_counter = stablehlo.add %counter, %step : tensor<i32>
        %new_sum = stablehlo.add %sum, %sum_step : tensor<i32>
        
        // Custom update for the third iteration argument
        %new_custom = "test.unknown_update"(%custom_val, %counter) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        %new_custom2 = "test.unknown_update"(%new_custom, %new_sum) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        
        stablehlo.return %new_counter, %new_sum, %new_custom2 : tensor<i32>, tensor<i32>, tensor<i32>
      }

  // Combine both results with the other value
  %combined = stablehlo.add %results#1, %results#2 : tensor<i32>
  %final_result = stablehlo.add %combined, %other_value : tensor<i32>
  
  return %final_result : tensor<i32>
}

// CHECK-LABEL: func.func @test_while_induction_2(
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<i32>) -> tensor<i32>
// CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK-DAG: %[[C3:.*]] = stablehlo.constant dense<3> : tensor<i32>
// CHECK-DAG: %[[C2:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK-DAG: %[[C8:.*]] = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT: %[[CUSTOM_INIT:.*]] = "test.unknown_state"() : () -> tensor<i32>

// CHECK: %[[WHILE:.*]]:2 = stablehlo.while(%[[ITER_CTR:.*]] = %[[C7]], %[[ITER_CUSTOM:.*]] = %[[CUSTOM_INIT]])
// CHECK: cond {
// CHECK:   %[[CMP:.*]] = stablehlo.compare LT, %[[ITER_CTR]], %[[ARG1]]
// CHECK:   stablehlo.return %[[CMP]] : tensor<i1>
// CHECK: } do {
// CHECK:   %[[OFFSET:.*]] = stablehlo.subtract %[[ITER_CTR]], %[[C7]]
// CHECK:   %[[SCALED:.*]] = stablehlo.multiply %[[OFFSET]], %[[C2]]
// CHECK:   %[[NEW_CTR:.*]] = stablehlo.add %[[ITER_CTR]], %[[C2]]
// CHECK:   %[[CALCULATED_SUM:.*]] = stablehlo.add %[[SCALED]], %[[C8]]
// CHECK:   %[[UPDATE1:.*]] = "test.unknown_update"(%[[ITER_CUSTOM]], %[[ITER_CTR]])
// CHECK:   %[[UPDATE2:.*]] = "test.unknown_update"(%[[UPDATE1]], %[[CALCULATED_SUM]])
// CHECK:   stablehlo.return %[[NEW_CTR]], %[[UPDATE2]]
// CHECK: }

// CHECK: %[[TOTAL_ITER:.*]] = stablehlo.subtract %[[ARG1]], %[[C7]]
// CHECK: %[[TOTAL_SCALED:.*]] = stablehlo.multiply %[[TOTAL_ITER]], %[[C2]]
// CHECK: %[[FINAL_SUM:.*]] = stablehlo.add %[[C3]], %[[TOTAL_SCALED]]
// CHECK: %[[COMBINED:.*]] = stablehlo.add %[[FINAL_SUM]], %[[WHILE]]#1
// CHECK: %[[RESULT:.*]] = stablehlo.add %[[COMBINED]], %[[ARG0]]
// CHECK: return %[[RESULT]]