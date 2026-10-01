// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(kernel-call-to-gpu-wrapper,llvm-to-memref-access,canonicalize)" | FileCheck %s

module {
  func.func private @square_kernel(%arg0: !llvm.ptr<1>) {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %index = arith.extui %tid : i32 to i64
    %ptr = llvm.getelementptr %arg0[%index]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %value = llvm.load %ptr : !llvm.ptr<1> -> f32
    %square = arith.mulf %value, %value : f32
    llvm.store %square, %ptr : f32, !llvm.ptr<1>
    return
  }

  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %result = enzymexla.kernel_call @square_kernel
        blocks in(%c1, %c1, %c1) threads in(%c4, %c1, %c1) shmem = %c0
        (%arg0) {
          output_operand_aliases = [#stablehlo.output_operand_alias<
              output_tuple_indices = [], operand_index = 0,
              operand_tuple_indices = []>]
        } : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func private @square_kernel$gpu_wrapper(
// CHECK-SAME:    %[[ARG:.*]]: memref<4xf32, 1>)
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         "enzymexla.gpu_wrapper"(%[[C1]], %[[C1]], %[[C1]], %[[C4]], %[[C1]], %[[C1]]) ({
// CHECK:           affine.parallel (%[[BX:.*]], %[[BY:.*]], %[[BZ:.*]], %[[TX:.*]], %[[TY:.*]], %[[TZ:.*]]) = (0, 0, 0, 0, 0, 0) to (1, 1, 1, 4, 1, 1) {
// CHECK:             %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[ARG]])
// CHECK:             %[[TID:.*]] = arith.index_castui %[[TX]] : index to i32
// CHECK:             llvm.getelementptr %[[PTR]]
// CHECK:           }
// CHECK:         })

// CHECK-LABEL: func.func @main
// CHECK:         %[[RESULT:.*]] = enzymexla.jit_call @square_kernel$gpu_wrapper (%arg0)
// CHECK-NOT:     enzymexla.kernel_call
// CHECK:         return %[[RESULT]] : tensor<4xf32>
