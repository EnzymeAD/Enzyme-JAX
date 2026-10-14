// RUN: enzymexlamlir-opt %s --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines

// Before: a guarded loop calls the wrapper inside an explicit allocation scope.
// After: the bound's stack slot is at the start of that scope. The store, device
// copy, and wrapper call stay inside the guard. Do not move the slot to the
// enclosing function or allocate it inside the guard.
module {
  func.func @nested_allocation_scope(%run: i1, %limit: i64, %data: memref<i32>) {
    memref.alloca_scope {
      scf.if %run {
        %zero = arith.constant 0 : i64
        %one = arith.constant 1 : i64
        scf.for %iv = %zero to %limit step %one : i64 {
          enzymexla.xla_wrapper @nested_body(%data) : (memref<i32>) -> ()
        }
      }
      memref.alloca_scope.return
    }
    return
  }

  func.func private @nested_body(%data: tensor<i32>) -> tensor<i32> {
    %negated = stablehlo.negate %data : tensor<i32>
    return %negated : tensor<i32>
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @nested_allocation_scope(%[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: memref<i32>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 8 : index
// CHECK-NEXT:     memref.alloca_scope  {
// CHECK-NEXT:       %[[ALLOCA_0:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT:       scf.if %[[VAL_0]] {
// CHECK-NEXT:         memref.store %[[VAL_1]], %[[ALLOCA_0]][] : memref<i64>
// CHECK-NEXT:         %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @nested_body_bound_1 : memref<i64, 1>
// CHECK-NEXT:         enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:         enzymexla.xla_wrapper @nested_body (%[[GET_GLOBAL_TEMP_0]], %[[VAL_2]]) : (memref<i64, 1>, memref<i32>) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @nested_body(%[[VAL_3:.*]]: tensor<i64>, %[[VAL_4:.*]]: tensor<i32>) -> (tensor<i64>, tensor<i32>) {
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_5:.*]] = %[[CONSTANT_1]], %[[VAL_6:.*]] = %[[VAL_4]]) : tensor<i64>, tensor<i32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_5]], %[[VAL_3]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_5]], %[[CONSTANT_2]] : tensor<i64>
// CHECK-NEXT:       %[[NEGATE_0:.*]] = stablehlo.negate %[[VAL_6]] : tensor<i32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[NEGATE_0]] : tensor<i64>, tensor<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_3]], %[[WHILE_0]]#1 : tensor<i64>, tensor<i32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @nested_body_bound_1 : memref<i64, 1>
// CHECK-NEXT: }
