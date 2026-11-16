// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU

// `k` is the number of elementary reflectors, i.e. the length of `tau`, which
// can be smaller than the number of columns

module {
  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<16xf32>) -> tensor<64x32xf32> {
      %0 = enzymexla.lapack.orgqr %arg0, %arg1 : (tensor<64x32xf32>, tensor<16xf32>) -> tensor<64x32xf32>
      return %0 : tensor<64x32xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sorgqr_[[WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// CPU-DAG:    %[[LAYOUT:.+]] = llvm.mlir.constant(102 : i64) : i64
// CPU-DAG:    %[[M:.+]] = llvm.mlir.constant(64 : i64) : i64
// CPU-DAG:    %[[N:.+]] = llvm.mlir.constant(32 : i64) : i64
// CPU-DAG:    %[[K:.+]] = llvm.mlir.constant(16 : i64) : i64
// CPU:    llvm.call @enzymexla_lapacke_sorgqr_(%[[LAYOUT]], %[[M]], %[[N]], %[[K]], %arg0, %[[M]], %arg1) : (i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
