// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines --implicit-check-not=scf.for

// A private function is not exclusive to the loop if another function calls it.
// Keep the original one-argument function for that caller. Give the lifted loop
// a separate function with a bound argument and a while loop.
module {
  func.func @lift_shared_body(%limit: i32, %data: memref<4xf32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      enzymexla.xla_wrapper @shared_body(%data) : (memref<4xf32>) -> ()
    }
    return
  }

  func.func @call_shared_body(%data: tensor<4xf32>) -> tensor<4xf32> {
    %result = func.call @shared_body(%data) : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func private @shared_body(%data: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.negate %data : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @lift_shared_body(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: memref<4xf32>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_0]], %[[ALLOCA_0]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @rxla$megakernel_0_bound_1 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     enzymexla.xla_wrapper @rxla$megakernel_0 (%[[GET_GLOBAL_TEMP_0]], %[[VAL_1]]) : (memref<i32, 1>, memref<4xf32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @call_shared_body(%[[VAL_2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %[[VAL_3:.*]] = call @shared_body(%[[VAL_2]]) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     return %[[VAL_3]] : tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @shared_body(%[[VAL_4:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %[[NEGATE_0:.*]] = stablehlo.negate %[[VAL_4]] : tensor<4xf32>
// CHECK-NEXT:     return %[[NEGATE_0]] : tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @rxla$megakernel_0(%[[VAL_5:.*]]: tensor<i32>, %[[VAL_6:.*]]: tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>) {
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_7:.*]] = %[[CONSTANT_1]], %[[VAL_8:.*]] = %[[VAL_6]]) : tensor<i32>, tensor<4xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_7]], %[[VAL_5]], SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_7]], %[[CONSTANT_2]] : tensor<i32>
// CHECK-NEXT:       %[[NEGATE_1:.*]] = stablehlo.negate %[[VAL_8]] : tensor<4xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[NEGATE_1]] : tensor<i32>, tensor<4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_5]], %[[WHILE_0]]#1 : tensor<i32>, tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @rxla$megakernel_0_bound_1 : memref<i32, 1>
// CHECK-NEXT: }

// -----

// A public function can have callers outside this module. Even with one local
// use, keep its signature and body. Put the loop in a new private function.
module {
  func.func @lift_public_body(%limit: i32, %data: memref<4xf32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      enzymexla.xla_wrapper @public_body(%data) : (memref<4xf32>) -> ()
    }
    return
  }

  func.func @public_body(%data: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.negate %data : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @lift_public_body(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: memref<4xf32>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_0]], %[[ALLOCA_0]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @rxla$megakernel_0_bound_1 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     enzymexla.xla_wrapper @rxla$megakernel_0 (%[[GET_GLOBAL_TEMP_0]], %[[VAL_1]]) : (memref<i32, 1>, memref<4xf32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @public_body(%[[VAL_2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %[[NEGATE_0:.*]] = stablehlo.negate %[[VAL_2]] : tensor<4xf32>
// CHECK-NEXT:     return %[[NEGATE_0]] : tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @rxla$megakernel_0(%[[VAL_3:.*]]: tensor<i32>, %[[VAL_4:.*]]: tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>) {
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_5:.*]] = %[[CONSTANT_1]], %[[VAL_6:.*]] = %[[VAL_4]]) : tensor<i32>, tensor<4xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_5]], %[[VAL_3]], SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_5]], %[[CONSTANT_2]] : tensor<i32>
// CHECK-NEXT:       %[[NEGATE_1:.*]] = stablehlo.negate %[[VAL_6]] : tensor<4xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[NEGATE_1]] : tensor<i32>, tensor<4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_3]], %[[WHILE_0]]#1 : tensor<i32>, tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @rxla$megakernel_0_bound_1 : memref<i32, 1>
// CHECK-NEXT: }

// -----

// A private function with only this wrapper use can change in place. Keep its
// name, add the bound argument/result, and replace its body with the while loop.
module {
  func.func @lift_exclusive_body(%limit: i32, %data: memref<4xf32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      enzymexla.xla_wrapper @exclusive_body(%data) : (memref<4xf32>) -> ()
    }
    return
  }

  func.func private @exclusive_body(%data: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.negate %data : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @lift_exclusive_body(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: memref<4xf32>) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %[[VAL_0]], %[[ALLOCA_0]][] : memref<i32>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @exclusive_body_bound_1 : memref<i32, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i32, 1>, memref<i32>
// CHECK-NEXT:     enzymexla.xla_wrapper @exclusive_body (%[[GET_GLOBAL_TEMP_0]], %[[VAL_1]]) : (memref<i32, 1>, memref<4xf32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @exclusive_body(%[[VAL_2:.*]]: tensor<i32>, %[[VAL_3:.*]]: tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>) {
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_4:.*]] = %[[CONSTANT_1]], %[[VAL_5:.*]] = %[[VAL_3]]) : tensor<i32>, tensor<4xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_4]], %[[VAL_2]], SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_4]], %[[CONSTANT_2]] : tensor<i32>
// CHECK-NEXT:       %[[NEGATE_0:.*]] = stablehlo.negate %[[VAL_5]] : tensor<4xf32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[NEGATE_0]] : tensor<i32>, tensor<4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_2]], %[[WHILE_0]]#1 : tensor<i32>, tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @exclusive_body_bound_1 : memref<i32, 1>
// CHECK-NEXT: }
