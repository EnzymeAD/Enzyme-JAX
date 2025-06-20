// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA

module {
  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
    %0:4 = enzymexla.linalg.svd %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sgesvd_[[WRAPPER_ID:[0-9]+]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(101 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(83 : i8) : i8
// CPU-NEXT:    %2 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %3 = llvm.call @enzymexla_lapacke_sgesvd_(%0, %1, %1, %2, %2, %arg0, %2, %arg2, %arg1, %2, %arg3, %2, %arg4) : (i64, i8, i8, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:    llvm.store %3, %arg5 : i64, !llvm.ptr
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sgesvd_(i64, i8, i8, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
// CPU-NEXT:    %[[u_vt:[a-z0-9_]+]] = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf32>
// CPU-NEXT:    %[[s:[a-z0-9_]+]] = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
// CPU-NEXT:    %[[superb:[a-z0-9_]+]] = stablehlo.constant dense<0.000000e+00> : tensor<63xf32>
// CPU-NEXT:    %[[info:[a-z0-9_]+]] = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %[[res:[a-z0-9_]+]]:4 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sgesvd_[[WRAPPER_ID]] (%arg0, %[[u_vt]], %[[s]], %[[u_vt]], %[[superb]], %[[info]]) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 3, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<63xf32>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
// CPU-NEXT:    return %[[res]]#0, %[[res]]#1, %[[res]]#2, %[[res]]#3 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>
// CPU-NEXT:  }

// CUDA: func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
// CUDA-NEXT:   %0:5 = stablehlo.custom_call @cusolver_gesvd_ffi(%arg0) {api_version = 4 : i32, compute_uv = true, full_matrices = false, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], transposed = false} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>)
// CUDA-NEXT:   return %0#2, %0#1, %0#3, %0#4 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>
// CUDA-NEXT: }

module {
  // CPU: func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xf64>, tensor<64x64xf64>, tensor<i64>) {
  func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xf64>, tensor<64x64xf64>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dgesvd_[[WRAPPER_ID:[0-9]+]]
    %0:4 = enzymexla.linalg.svd %arg0 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xf64>, tensor<64x64xf64>, tensor<i64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xf64>, tensor<64xf64>, tensor<64x64xf64>, tensor<i64>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xf32>, tensor<64x64xcomplex<f32>>, tensor<i64>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xf32>, tensor<64x64xcomplex<f32>>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cgesvd_[[WRAPPER_ID:[0-9]+]]
    %0:4 = enzymexla.linalg.svd %arg0 : (tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xf32>, tensor<64x64xcomplex<f32>>, tensor<i64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xcomplex<f32>>, tensor<64xf32>, tensor<64x64xcomplex<f32>>, tensor<i64>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xf64>, tensor<64x64xcomplex<f64>>, tensor<i64>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xf64>, tensor<64x64xcomplex<f64>>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zgesvd_[[WRAPPER_ID:[0-9]+]]
    %0:4 = enzymexla.linalg.svd %arg0 : (tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xf64>, tensor<64x64xcomplex<f64>>, tensor<i64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xcomplex<f64>>, tensor<64xf64>, tensor<64x64xcomplex<f64>>, tensor<i64>
  }
}
