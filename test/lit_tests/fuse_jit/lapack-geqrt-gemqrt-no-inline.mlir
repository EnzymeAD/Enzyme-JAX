// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(fuse-jit)" %s | FileCheck %s

// GEQRT -> GEMQRT; test direct fuse-jit behavior without inlining.

module {
  llvm.func @enzymexla_wrapper_lapacke_dgeqrt_1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.mlir.constant(102 : i64) : i64
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.call @enzymexla_lapacke_dgeqrt_(%0, %1, %1, %1, %arg0, %1, %arg1, %1) : (i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
    llvm.store %2, %arg2 : i64, !llvm.ptr
    llvm.return
  }
  llvm.func @enzymexla_lapacke_dgeqrt_(i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
  llvm.func @enzymexla_wrapper_lapacke_dgemqrt_1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.mlir.constant(102 : i64) : i64
    %1 = llvm.mlir.constant(76 : i8) : i8
    %2 = llvm.mlir.constant(78 : i8) : i8
    %3 = llvm.mlir.constant(4 : i64) : i64
    %4 = llvm.call @enzymexla_lapacke_dgemqrt_(%0, %1, %2, %3, %3, %3, %3, %arg0, %3, %arg1, %3, %arg2, %3) : (i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
    llvm.return
  }
  llvm.func @enzymexla_lapacke_dgemqrt_(i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
  func.func @lapack_compact_qr_chain(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %c = stablehlo.constant dense<-1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf64>
    %0:3 = enzymexla.jit_call @enzymexla_wrapper_lapacke_dgeqrt_1 (%arg0, %cst, %c) <operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<4x4xf64>, tensor<4x4xf64>, tensor<i64>) -> (tensor<4x4xf64>, tensor<4x4xf64>, tensor<i64>)
    %1 = enzymexla.jit_call @enzymexla_wrapper_lapacke_dgemqrt_1 (%0#0, %0#1, %arg1) <operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<4x4xf64>, tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
    return %1 : tensor<4x4xf64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_lapacke_dgeqrt_1_enzymexla_wrapper_lapacke_dgemqrt_1(
// CHECK: llvm.call @enzymexla_lapacke_dgeqrt_(
// CHECK: llvm.call @enzymexla_lapacke_dgemqrt_(
// CHECK: llvm.return
// CHECK-LABEL: func.func @lapack_compact_qr_chain(
// CHECK-NOT: enzymexla.jit_call @enzymexla_
// CHECK: enzymexla.jit_call @fused__enzymexla_wrapper_lapacke_dgeqrt_1_enzymexla_wrapper_lapacke_dgemqrt_1
// CHECK-SAME: operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>]
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3
// CHECK-SAME: output_tuple_indices = [1], operand_index = 0
// CHECK-SAME: output_tuple_indices = [2], operand_index = 1
// CHECK-SAME: output_tuple_indices = [3], operand_index = 2
