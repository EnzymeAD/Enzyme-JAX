// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(fuse-jit)" %s | FileCheck %s

// GEQRF -> ORGQR -> POTRF; test direct fuse-jit behavior without inlining.

module {
  llvm.func @enzymexla_wrapper_lapacke_dgeqrf_2(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.mlir.constant(102 : i64) : i64
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.call @enzymexla_lapacke_dgeqrf_(%0, %1, %1, %arg0, %1, %arg1) : (i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
    llvm.store %2, %arg2 : i64, !llvm.ptr
    llvm.return
  }
  llvm.func @enzymexla_lapacke_dgeqrf_(i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
  llvm.func @enzymexla_wrapper_lapacke_dorgqr_2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.constant(102 : i64) : i64
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.call @enzymexla_lapacke_dorgqr_(%0, %1, %1, %1, %arg0, %1, %arg1) : (i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
    llvm.return
  }
  llvm.func @enzymexla_lapacke_dorgqr_(i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
  llvm.func @enzymexla_wrapper_lapack_dpotrf_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) {
    llvm.call @enzymexla_lapack_dpotrf_(%arg0, %arg1, %arg2, %arg3, %arg4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @enzymexla_lapack_dpotrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func @lapack_chain(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<i64>) {
    %c = stablehlo.constant dense<4> : tensor<i64>
    %c_0 = stablehlo.constant dense<85> : tensor<i8>
    %c_1 = stablehlo.constant dense<-1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %0:3 = enzymexla.jit_call @enzymexla_wrapper_lapacke_dgeqrf_2 (%arg0, %cst, %c_1) <operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>)
    %1 = enzymexla.jit_call @enzymexla_wrapper_lapacke_dorgqr_2 (%0#0, %0#1) <operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4x4xf64>
    %2:2 = enzymexla.jit_call @enzymexla_wrapper_lapack_dpotrf_ (%c_0, %c, %1, %c, %c_1) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i8>, tensor<i64>, tensor<4x4xf64>, tensor<i64>, tensor<i64>) -> (tensor<4x4xf64>, tensor<i64>)
    return %2#0, %2#1 : tensor<4x4xf64>, tensor<i64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_lapacke_dgeqrf_2_enzymexla_wrapper_lapacke_dorgqr_2_enzymexla_wrapper_lapack_dpotrf_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr) {
// CHECK-NEXT: %[[LAYOUT:.*]] = llvm.mlir.constant(102 : i64) : i64
// CHECK-NEXT: %[[N:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[INFO:.*]] = llvm.call @enzymexla_lapacke_dgeqrf_(%[[LAYOUT]], %[[N]], %[[N]], %arg0, %[[N]], %arg1) : (i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CHECK-NEXT: llvm.store %[[INFO]], %arg2 : i64, !llvm.ptr
// CHECK-NEXT: %[[QINFO:.*]] = llvm.call @enzymexla_lapacke_dorgqr_(%[[LAYOUT]], %[[N]], %[[N]], %[[N]], %arg0, %[[N]], %arg1) : (i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CHECK-NEXT: llvm.call @enzymexla_lapack_dpotrf_(%arg3, %arg4, %arg0, %arg5, %arg6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT: llvm.return
// CHECK-LABEL: func.func @lapack_chain(
// CHECK-NOT: enzymexla.jit_call @enzymexla_
// CHECK: enzymexla.jit_call @fused__enzymexla_wrapper_lapacke_dgeqrf_2_enzymexla_wrapper_lapacke_dorgqr_2_enzymexla_wrapper_lapack_dpotrf_
// CHECK-SAME: operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>]
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0
// CHECK-SAME: output_tuple_indices = [1], operand_index = 6
// CHECK-SAME: output_tuple_indices = [2], operand_index = 1
// CHECK-SAME: output_tuple_indices = [3], operand_index = 2
