// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x48xf32>) -> tensor<64x48xf32> {
    %0 = enzymexla.lapack.gemqrt %arg0, %arg1, %arg2 {side = #enzymexla.side<left>} : (tensor<64x32xf32>, tensor<32x32xf32>, tensor<64x48xf32>) -> tensor<64x48xf32>
    return %0: tensor<64x48xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sgemqrt_[[WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(102 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(76 : i8) : i8
// CPU-NEXT:    %2 = llvm.mlir.constant(78 : i8) : i8
// CPU-NEXT:    %3 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %4 = llvm.mlir.constant(48 : i64) : i64
// CPU-NEXT:    %5 = llvm.mlir.constant(32 : i64) : i64
// CPU-NEXT:    %6 = llvm.call @enzymexla_lapacke_sgemqrt_(%0, %1, %2, %3, %4, %5, %5, %arg0, %3, %arg1, %5, %arg2, %3) : (i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sgemqrt_(i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x48xf32>) -> tensor<64x48xf32> {
// CPU-NEXT:    %0 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sgemqrt_[[WRAPPER_ID]] (%arg0, %arg1, %arg2) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<64x32xf32>, tensor<32x32xf32>, tensor<64x48xf32>) -> tensor<64x48xf32>
// CPU-NEXT:    return %0 : tensor<64x48xf32>
// CPU-NEXT:  }

module {
  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x48xf32>) -> tensor<64x48xf32> {
    %0 = enzymexla.lapack.gemqrt %arg0, %arg1, %arg2 {side = #enzymexla.side<left>, transpose = #enzymexla.transpose<transpose>} : (tensor<64x32xf32>, tensor<32x32xf32>, tensor<64x48xf32>) -> tensor<64x48xf32>
    return %0: tensor<64x48xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sgemqrt_[[WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(102 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(76 : i8) : i8
// CPU-NEXT:    %2 = llvm.mlir.constant(84 : i8) : i8
// CPU-NEXT:    %3 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %4 = llvm.mlir.constant(48 : i64) : i64
// CPU-NEXT:    %5 = llvm.mlir.constant(32 : i64) : i64
// CPU-NEXT:    %6 = llvm.call @enzymexla_lapacke_sgemqrt_(%0, %1, %2, %3, %4, %5, %5, %arg0, %3, %arg1, %5, %arg2, %3) : (i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sgemqrt_(i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x48xf32>) -> tensor<64x48xf32> {
// CPU-NEXT:    %0 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sgemqrt_[[WRAPPER_ID]] (%arg0, %arg1, %arg2) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<64x32xf32>, tensor<32x32xf32>, tensor<64x48xf32>) -> tensor<64x48xf32>
// CPU-NEXT:    return %0 : tensor<64x48xf32>
// CPU-NEXT:  }

module {
  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<48x64xf32>) -> tensor<48x64xf32> {
    %0 = enzymexla.lapack.gemqrt %arg0, %arg1, %arg2 {side = #enzymexla.side<right>} : (tensor<64x32xf32>, tensor<32x32xf32>, tensor<48x64xf32>) -> tensor<48x64xf32>
    return %0: tensor<48x64xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sgemqrt_[[WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(102 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(82 : i8) : i8
// CPU-NEXT:    %2 = llvm.mlir.constant(78 : i8) : i8
// CPU-NEXT:    %3 = llvm.mlir.constant(48 : i64) : i64
// CPU-NEXT:    %4 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %5 = llvm.mlir.constant(32 : i64) : i64
// CPU-NEXT:    %6 = llvm.call @enzymexla_lapacke_sgemqrt_(%0, %1, %2, %3, %4, %5, %5, %arg0, %4, %arg1, %5, %arg2, %3) : (i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sgemqrt_(i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<48x64xf32>) -> tensor<48x64xf32> {
// CPU-NEXT:    %0 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sgemqrt_[[WRAPPER_ID]] (%arg0, %arg1, %arg2) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<64x32xf32>, tensor<32x32xf32>, tensor<48x64xf32>) -> tensor<48x64xf32>
// CPU-NEXT:    return %0 : tensor<48x64xf32>
// CPU-NEXT:  }

module {
  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<48x64xf32>) -> tensor<48x64xf32> {
    %0 = enzymexla.lapack.gemqrt %arg0, %arg1, %arg2 {side = #enzymexla.side<right>, transpose = #enzymexla.transpose<transpose>} : (tensor<64x32xf32>, tensor<32x32xf32>, tensor<48x64xf32>) -> tensor<48x64xf32>
    return %0: tensor<48x64xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sgemqrt_[[WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(102 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(82 : i8) : i8
// CPU-NEXT:    %2 = llvm.mlir.constant(84 : i8) : i8
// CPU-NEXT:    %3 = llvm.mlir.constant(48 : i64) : i64
// CPU-NEXT:    %4 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %5 = llvm.mlir.constant(32 : i64) : i64
// CPU-NEXT:    %6 = llvm.call @enzymexla_lapacke_sgemqrt_(%0, %1, %2, %3, %4, %5, %5, %arg0, %4, %arg1, %5, %arg2, %3) : (i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sgemqrt_(i64, i8, i8, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<48x64xf32>) -> tensor<48x64xf32> {
// CPU-NEXT:    %0 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sgemqrt_[[WRAPPER_ID]] (%arg0, %arg1, %arg2) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], xla_side_effect_free} : (tensor<64x32xf32>, tensor<32x32xf32>, tensor<48x64xf32>) -> tensor<48x64xf32>
// CPU-NEXT:    return %0 : tensor<48x64xf32>
// CPU-NEXT:  }

module {
  func.func @main(%arg0: tensor<64x32xf64>, %arg1: tensor<32x32xf64>, %arg2: tensor<64x48xf64>) -> tensor<64x48xf64> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dgemqrt_[[WRAPPER_ID:[0-9]+]]
    %0 = enzymexla.lapack.gemqrt %arg0, %arg1, %arg2 {side = #enzymexla.side<left>} : (tensor<64x32xf64>, tensor<32x32xf64>, tensor<64x48xf64>) -> tensor<64x48xf64>
    return %0: tensor<64x48xf64>
  }
}

module {
  func.func @main(%arg0: tensor<64x32xcomplex<f32>>, %arg1: tensor<32x32xcomplex<f32>>, %arg2: tensor<64x48xcomplex<f32>>) -> tensor<64x48xcomplex<f32>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cgemqrt_[[WRAPPER_ID:[0-9]+]]
    %0 = enzymexla.lapack.gemqrt %arg0, %arg1, %arg2 {side = #enzymexla.side<left>} : (tensor<64x32xcomplex<f32>>, tensor<32x32xcomplex<f32>>, tensor<64x48xcomplex<f32>>) -> tensor<64x48xcomplex<f32>>
    return %0: tensor<64x48xcomplex<f32>>
  }
}

module {
  func.func @main(%arg0: tensor<64x32xcomplex<f64>>, %arg1: tensor<32x32xcomplex<f64>>, %arg2: tensor<64x48xcomplex<f64>>) -> tensor<64x48xcomplex<f64>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zgemqrt_[[WRAPPER_ID:[0-9]+]]
    %0 = enzymexla.lapack.gemqrt %arg0, %arg1, %arg2 {side = #enzymexla.side<left>} : (tensor<64x32xcomplex<f64>>, tensor<32x32xcomplex<f64>>, tensor<64x48xcomplex<f64>>) -> tensor<64x48xcomplex<f64>>
    return %0: tensor<64x48xcomplex<f64>>
  }
}
