// RUN: enzymexlamlir-opt %s --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines --implicit-check-not=scf.for --implicit-check-not=gpu.alloc --implicit-check-not=gpu.dealloc

// Fuse two calls, then lift their loop. Keep the limit, step, and specialized
// scalar outside the while state. Put stack slots in the function entry block.
// Keep stores and copies inside the host guard.
module {
  func.func @guarded_specialized_loop(%limit: i64, %step: i64,
                                     %data: memref<i32>, %scale: i32) {
    %zero = arith.constant 0 : i64
    %positive = arith.cmpi sgt, %step, %zero : i64
    scf.if %positive {
      scf.for %iv = %zero to %limit step %step : i64 {
        enzymexla.xla_wrapper @add_scale(%data, %scale)
            <{num_specialized = 1 : i64}> : (memref<i32>, i32) -> ()
        enzymexla.xla_wrapper @negate(%data) : (memref<i32>) -> ()
      }
    }
    return
  }

  func.func private @add_scale(%data: tensor<i32>, %scale: tensor<i32>)
      -> tensor<i32> {
    %sum = stablehlo.add %data, %scale : tensor<i32>
    return %sum : tensor<i32>
  }

  func.func private @negate(%data: tensor<i32>) -> tensor<i32> {
    %negated = stablehlo.negate %data : tensor<i32>
    return %negated : tensor<i32>
  }
}

// The specialized scalar is the last input. Only the bound buffers and updated
// data have results. The bound buffers return their original values.

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @guarded_specialized_loop(%[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: memref<i32>, %[[VAL_3:.*]]: i32) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 8 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : i64
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT:     %[[ALLOCA_1:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT:     %[[CMPI_0:.*]] = arith.cmpi sgt, %[[VAL_1]], %[[CONSTANT_1]] : i64
// CHECK-NEXT:     scf.if %[[CMPI_0]] {
// CHECK-NEXT:       memref.store %[[VAL_0]], %[[ALLOCA_1]][] : memref<i64>
// CHECK-NEXT:       %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @add_scale_bound_1 : memref<i64, 1>
// CHECK-NEXT:       enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_1]], %[[CONSTANT_0]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:       memref.store %[[VAL_1]], %[[ALLOCA_0]][] : memref<i64>
// CHECK-NEXT:       %[[GET_GLOBAL_TEMP_1:.*]] = enzymexla.get_global_temp @add_scale_bound_2 : memref<i64, 1>
// CHECK-NEXT:       enzymexla.memcpy  %[[GET_GLOBAL_TEMP_1]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:       enzymexla.xla_wrapper @add_scale (%[[GET_GLOBAL_TEMP_0]], %[[GET_GLOBAL_TEMP_1]], %[[VAL_2]], %[[VAL_3]]) <num_specialized = 1> : (memref<i64, 1>, memref<i64, 1>, memref<i32>, i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @add_scale(%[[VAL_4:.*]]: tensor<i64>, %[[VAL_5:.*]]: tensor<i64>, %[[VAL_6:.*]]: tensor<i32>, %[[VAL_7:.*]]: tensor<i32>) -> (tensor<i64>, tensor<i64>, tensor<i32>) {
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_8:.*]] = %[[CONSTANT_2]], %[[VAL_9:.*]] = %[[VAL_6]]) : tensor<i64>, tensor<i32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_8]], %[[VAL_4]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_8]], %[[VAL_5]] : tensor<i64>
// CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %[[VAL_9]], %[[VAL_7]] : tensor<i32>
// CHECK-NEXT:       %[[NEGATE_0:.*]] = stablehlo.negate %[[ADD_1]] : tensor<i32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[NEGATE_0]] : tensor<i64>, tensor<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_4]], %[[VAL_5]], %[[WHILE_0]]#1 : tensor<i64>, tensor<i64>, tensor<i32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @add_scale_bound_1 : memref<i64, 1>
// CHECK-NEXT:   enzymexla.temp_alloc "private" @add_scale_bound_2 : memref<i64, 1>
// CHECK-NEXT: }
