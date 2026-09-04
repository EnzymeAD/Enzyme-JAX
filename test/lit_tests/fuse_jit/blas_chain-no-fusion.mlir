// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-blas{backend=cpu blas_int_width=64},fuse-jit)" %s | FileCheck %s

// CPU BLAS lowering places each real layout-bearing JIT in a generated helper.
// A caller-level BLAS chain must keep those helpers and JITs intact.

module {
  func.func @main(%a: tensor<64x64xf32>, %b: tensor<64x32xf32>, %c: tensor<64x32xf32>) -> tensor<64x32xf32> {
    %alpha = stablehlo.constant dense<2.0> : tensor<f32>
    %beta = stablehlo.constant dense<3.0> : tensor<f32>
    %first = enzymexla.blas.symm %a, %b, %c, %alpha, %beta {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<64x32xf32>, tensor<f32>, tensor<f32>) -> tensor<64x32xf32>
    %second = enzymexla.blas.symm %a, %b, %first, %alpha, %beta {side = #enzymexla.side<left>, uplo = #enzymexla.uplo<U>} : (tensor<64x64xf32>, tensor<64x32xf32>, tensor<64x32xf32>, tensor<f32>, tensor<f32>) -> tensor<64x32xf32>
    return %second : tensor<64x32xf32>
  }
}

// CHECK-NOT: llvm.func @fused__
// CHECK:       enzymexla.jit_call @enzymexla_blas_ssymm_wrapper
// CHECK-SAME:  operand_layouts =
// CHECK-SAME:  result_layouts =
// CHECK:       enzymexla.jit_call @enzymexla_blas_ssymm_wrapper
// CHECK-SAME:  operand_layouts =
// CHECK-SAME:  result_layouts =
// CHECK-NOT:   llvm.func @fused__

// CHECK-LABEL: func.func @main
// CHECK:         %[[FIRST:.*]] = call @enzymexla_blas_ssymm_wrapper_{{[0-9]+}}(%arg0, %arg1, %arg2, %{{.*}}, %{{.*}})
// CHECK:         %[[SECOND:.*]] = call @enzymexla_blas_ssymm_wrapper_{{[0-9]+}}(%arg0, %arg1, %[[FIRST]], %{{.*}}, %{{.*}})
// CHECK-NOT:     enzymexla.jit_call @fused__
// CHECK:         return %[[SECOND]] : tensor<64x32xf32>
