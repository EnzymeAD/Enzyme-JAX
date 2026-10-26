// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(inline{default-pipeline=canonicalize max-iterations=4},fuse-jit)" %s | FileCheck %s

// SYMM -> SYRK; inline generated helpers so fuse-jit can see the call chain.

module {
  func.func private @enzymexla_blas_dsyrk_wrapper_0(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> tensor<4x4xf64> {
    %c = stablehlo.constant dense<84> : tensor<ui8>
    %c_0 = stablehlo.constant dense<76> : tensor<ui8>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %0 = enzymexla.jit_call @enzymexla_blas_dsyrk_wrapper (%c_0, %c, %c_1, %c_1, %arg2, %arg0, %c_1, %arg3, %arg1, %c_1) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 8, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<ui8>, tensor<ui8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<4x4xf64>, tensor<i64>, tensor<f64>, tensor<4x4xf64>, tensor<i64>) -> tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
  }

  llvm.func private @enzymexla_blas_dsyrk_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @enzymexla_blas_dsyrk_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %0, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }

  llvm.func @enzymexla_blas_dsyrk_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)

  func.func private @enzymexla_blas_dsymm_wrapper_0(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>, %arg2: tensor<4x4xf64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> tensor<4x4xf64> {
    %c = stablehlo.constant dense<82> : tensor<ui8>
    %c_0 = stablehlo.constant dense<76> : tensor<ui8>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %0 = enzymexla.jit_call @enzymexla_blas_dsymm_wrapper (%c, %c_0, %c_1, %c_1, %arg3, %arg0, %c_1, %arg1, %c_1, %arg4, %arg2, %c_1) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 10, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<ui8>, tensor<ui8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<4x4xf64>, tensor<i64>, tensor<4x4xf64>, tensor<i64>, tensor<f64>, tensor<4x4xf64>, tensor<i64>) -> tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
  }

  llvm.func private @enzymexla_blas_dsymm_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @enzymexla_blas_dsymm_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %0, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }

  llvm.func @enzymexla_blas_dsymm_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)

  func.func @blas_chain(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>, %arg2: tensor<4x4xf64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> tensor<4x4xf64> {
    %0 = call @enzymexla_blas_dsymm_wrapper_0(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<4x4xf64>, tensor<4x4xf64>, tensor<4x4xf64>, tensor<f64>, tensor<f64>) -> tensor<4x4xf64>
    %1 = call @enzymexla_blas_dsyrk_wrapper_0(%arg0, %0, %arg3, %arg4) : (tensor<4x4xf64>, tensor<4x4xf64>, tensor<f64>, tensor<f64>) -> tensor<4x4xf64>
    return %1 : tensor<4x4xf64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_blas_dsymm_wrapper_enzymexla_blas_dsyrk_wrapper(
// CHECK: llvm.call @enzymexla_blas_dsymm_(
// CHECK: llvm.call @enzymexla_blas_dsyrk_(
// CHECK: llvm.return
// CHECK-LABEL: func.func @blas_chain(
// CHECK-NOT: enzymexla.jit_call @enzymexla_
// CHECK: enzymexla.jit_call @fused__enzymexla_blas_dsymm_wrapper_enzymexla_blas_dsyrk_wrapper
// CHECK-SAME: operand_layouts = [dense<> : tensor<0xindex>, {{.*}}dense<[1, 0]> : tensor<2xindex>
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 10
// CHECK-NOT: enzymexla.jit_call @enzymexla_
