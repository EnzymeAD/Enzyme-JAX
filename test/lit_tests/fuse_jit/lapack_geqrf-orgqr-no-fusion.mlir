// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},fuse-jit)" %s | FileCheck %s

// geqrf -> orgqr is a direct real JIT dependency, but CPU LAPACK calls carry
// layout metadata which the current fusion pass deliberately rejects.

module {
  func.func @main(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %qr:3 = enzymexla.lapack.geqrf %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<i64>)
    %q = enzymexla.lapack.orgqr %qr#0, %qr#1 : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
    return %q : tensor<64x64xf32>
  }
}

// CHECK-NOT: llvm.func @fused__
// CHECK-LABEL: func.func @main
// CHECK:         %[[GEQRF:.*]]:3 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sgeqrf_{{[0-9]+}}
// CHECK-SAME:    operand_layouts =
// CHECK-SAME:    result_layouts =
// CHECK:         %[[ORGQR:.*]] = enzymexla.jit_call @enzymexla_wrapper_lapacke_sorgqr_{{[0-9]+}} (%[[GEQRF]]#0, %[[GEQRF]]#1)
// CHECK-SAME:    operand_layouts =
// CHECK-SAME:    result_layouts =
// CHECK-NOT:     enzymexla.jit_call @fused__
// CHECK:         return %[[ORGQR]] : tensor<64x64xf32>
