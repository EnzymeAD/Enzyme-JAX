// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(inline{default-pipeline=canonicalize max-iterations=4},fuse-jit)" %s | FileCheck %s

// GESVD -> GESVD; inline generated helpers so fuse-jit can see the call chain.

module {
  func.func private @shlo_enzymexla_wrapper_lapack_dgesvd__wrapper_1(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf64>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %0:5 = enzymexla.jit_call @enzymexla_wrapper_lapack_dgesvd_ (%c_1, %c_1, %arg0, %c_1, %cst, %cst_0, %c_1, %cst_0, %c_1, %c) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 9, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 2, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<4x4xf64>, tensor<i64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>, tensor<4x4xf64>, tensor<i64>, tensor<i64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>, tensor<4x4xf64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>
  }

  func.func private @shlo_enzymexla_wrapper_lapack_dgesvd__wrapper_0(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf64>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %0:5 = enzymexla.jit_call @enzymexla_wrapper_lapack_dgesvd_ (%c_1, %c_1, %arg0, %c_1, %cst, %cst_0, %c_1, %cst_0, %c_1, %c) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 9, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 2, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<4x4xf64>, tensor<i64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>, tensor<4x4xf64>, tensor<i64>, tensor<i64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>, tensor<4x4xf64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>
  }

  llvm.func private @enzymexla_wrapper_lapack_dgesvd_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
    %0 = llvm.mlir.constant(83 : i8) : i8
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(-1 : i64) : i64
    %3 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
    llvm.store %2, %3 : i64, !llvm.ptr
    %4 = llvm.alloca %1 x f64 : (i64) -> !llvm.ptr
    %5 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
    %6 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
    llvm.store %0, %5 : i8, !llvm.ptr
    llvm.store %0, %6 : i8, !llvm.ptr
    llvm.call @enzymexla_lapack_dgesvd_(%5, %6, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %4, %3, %arg9, %1, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    %7 = llvm.load %4 : !llvm.ptr -> f64
    %8 = llvm.fptosi %7 : f64 to i64
    %9 = llvm.alloca %8 x f64 : (i64) -> !llvm.ptr
    llvm.store %8, %3 : i64, !llvm.ptr
    llvm.call @enzymexla_lapack_dgesvd_(%5, %6, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %9, %3, %arg9, %1, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }

  llvm.func @enzymexla_lapack_dgesvd_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)

  func.func @lapack_gesvd_chain(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>) {
    %0:4 = call @shlo_enzymexla_wrapper_lapack_dgesvd__wrapper_1(%arg0) : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>)
    %1:4 = call @shlo_enzymexla_wrapper_lapack_dgesvd__wrapper_0(%0#0) : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>)
    return %1#0, %1#1, %1#2, %1#3 : tensor<4x4xf64>, tensor<4xf64>, tensor<4x4xf64>, tensor<i64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_lapack_dgesvd__enzymexla_wrapper_lapack_dgesvd_(
// CHECK-COUNT-4: llvm.call @enzymexla_lapack_dgesvd_(
// CHECK: llvm.return
// CHECK-LABEL: func.func @lapack_gesvd_chain(
// CHECK-NOT: enzymexla.jit_call @enzymexla_
// CHECK: enzymexla.jit_call @fused__enzymexla_wrapper_lapack_dgesvd__enzymexla_wrapper_lapack_dgesvd_
// CHECK-SAME: operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 14
// CHECK-SAME: #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 13
// CHECK-NOT: enzymexla.jit_call @enzymexla_
