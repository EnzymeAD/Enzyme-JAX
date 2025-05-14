// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>) {
    %0:3 = enzymexla.linalg.lu %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64xi32>, tensor<i32>
  }
}

// CPU:  llvm.func @enzymexla_lapack_sgetrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU-NEXT:  llvm.func @enzymexla_lapack_sgetrf_wrapper_3(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:      %0 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:      %1 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:      %2 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:      %3 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:      llvm.store %0, %2 : i64, !llvm.ptr
// CPU-NEXT:      llvm.store %0, %3 : i64, !llvm.ptr
// CPU-NEXT:      llvm.call @enzymexla_lapack_sgetrf_(%2, %3, %arg0, %2, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:      llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>) {
// CPU-NEXT:      %c = stablehlo.constant dense<-1> : tensor<64xi64>
// CPU-NEXT:      %c_0 = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:      %0:3 = enzymexla.jit_call @enzymexla_lapack_sgetrf_wrapper_3 (%arg0, %c, %c_0) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>)
// CPU-NEXT:      %1 = stablehlo.convert %0#1 : (tensor<64xi64>) -> tensor<64xi32>
// CPU-NEXT:      %2 = stablehlo.convert %0#2 : (tensor<i64>) -> tensor<i32>
// CPU-NEXT:      return %0#0, %1, %2 : tensor<64x64xf32>, tensor<64xi32>, tensor<i32>
// CPU-NEXT:  }

// CUDA: func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>) {
// CUDA-NEXT:     %0:3 = stablehlo.custom_call @cusolver_getrf_ffi(%arg0) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>)
// CUDA-NEXT:     return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64xi32>, tensor<i32>
// CUDA-NEXT: }

// TPU: func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>) {
// TPU-NEXT:     %c = stablehlo.constant dense<true> : tensor<i1>
// TPU-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<64xi32>
// TPU-NEXT:     %0:3 = stablehlo.custom_call @LUFactorization(%arg0) : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>)
// TPU-NEXT:     %1 = stablehlo.add %c_0, %0#1 : tensor<64xi32>
// TPU-NEXT:     %2 = stablehlo.is_finite %0#0 : (tensor<64x64xf32>) -> tensor<64x64xi1>
// TPU-NEXT:     %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.and across dimensions = [0, 1] : (tensor<64x64xi1>, tensor<i1>) -> tensor<i1>
// TPU-NEXT:     %4 = stablehlo.not %3 : tensor<i1>
// TPU-NEXT:     %5 = stablehlo.convert %4 : (tensor<i1>) -> tensor<i32>
// TPU-NEXT:     return %0#0, %1, %5 : tensor<64x64xf32>, tensor<64xi32>, tensor<i32>
// TPU-NEXT: }

module {
  // CPU: func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<i32>) {
  func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<i32>) {
    // CPU: enzymexla.jit_call @enzymexla_lapack_dgetrf_wrapper_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.linalg.lu %arg0 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#2 : tensor<64x64xf64>, tensor<64xi32>, tensor<i32>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>) {
    // CPU: enzymexla.jit_call @enzymexla_lapack_zgetrf_wrapper_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.linalg.lu %arg0 : (tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#2 : tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>) {
    // CPU: enzymexla.jit_call @enzymexla_lapack_cgetrf_wrapper_[[WRAPPER_ID:[0-9]+]]
    %0:3 = enzymexla.linalg.lu %arg0 : (tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#2 : tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>
  }
}
