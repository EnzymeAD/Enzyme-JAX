// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>) {
    %0:4 = enzymexla.linalg.lu %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>
  }
}

// CPU:  func.func private @enzymexla_lapack_sgetrf_[[WRAPPER_ID:[0-9]+]](%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>) {
// CPU-NEXT:    %c = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<-1> : tensor<64xi64>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:    %0:3 = enzymexla.jit_call @enzymexla_lapack_sgetrf_wrapper (%c_1, %c_1, %arg0, %c_1, %c_0, %c) {operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<i64>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<64xi64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>)
// CPU-NEXT:    stablehlo.return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64xi64>, tensor<i64>
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func private @enzymexla_lapack_sgetrf_wrapper(%arg0: !llvm.ptr {llvm.nofree, llvm.readonly}, %arg1: !llvm.ptr {llvm.nofree, llvm.readonly}, %arg2: !llvm.ptr {llvm.nofree}, %arg3: !llvm.ptr {llvm.nofree, llvm.readonly}, %arg4: !llvm.ptr {llvm.nofree, llvm.writeonly}, %arg5: !llvm.ptr {llvm.nofree, llvm.writeonly}) {
// CPU-NEXT:    llvm.call @enzymexla_lapack_sgetrf_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapack_sgetrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU-NEXT:  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>) {
// CPU-NEXT:    %c = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<64> : tensor<i32>
// CPU-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<64xi64>
// CPU-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<i32>
// CPU-NEXT:    %0:3 = call @enzymexla_lapack_sgetrf_[[WRAPPER_ID]](%arg0) : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>)
// CPU-NEXT:    %1 = stablehlo.subtract %0#1, %c_2 : tensor<64xi64>
// CPU-NEXT:    %2:2 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %c) : tensor<i32>, tensor<64xi64>
// CPU-NEXT:    cond {
// CPU-NEXT:      %7 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CPU-NEXT:      stablehlo.return %7 : tensor<i1>
// CPU-NEXT:    } do {
// CPU-NEXT:      %7 = stablehlo.add %iterArg, %c_0 : tensor<i32>
// CPU-NEXT:      %8 = stablehlo.dynamic_slice %1, %iterArg, sizes = [1] : (tensor<64xi64>, tensor<i32>) -> tensor<1xi64>
// CPU-NEXT:      %9 = stablehlo.dynamic_slice %iterArg_4, %iterArg, sizes = [1] : (tensor<64xi64>, tensor<i32>) -> tensor<1xi64>
// CPU-NEXT:      %10 = "stablehlo.gather"(%iterArg_4, %8) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], start_index_map = [0]>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<64xi64>, tensor<1xi64>) -> tensor<1xi64>
// CPU-NEXT:      %11 = stablehlo.dynamic_update_slice %iterArg_4, %10, %iterArg : (tensor<64xi64>, tensor<1xi64>, tensor<i32>) -> tensor<64xi64>
// CPU-NEXT:      %12 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
// CPU-NEXT:      %13 = "stablehlo.scatter"(%11, %8, %12) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
// CPU-NEXT:      ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
// CPU-NEXT:        stablehlo.return %arg2 : tensor<i64>
// CPU-NEXT:      }) : (tensor<64xi64>, tensor<1xi64>, tensor<i64>) -> tensor<64xi64>
// CPU-NEXT:      stablehlo.return %7, %13 : tensor<i32>, tensor<64xi64>
// CPU-NEXT:    }
// CPU-NEXT:    %3 = stablehlo.add %2#1, %c_2 : tensor<64xi64>
// CPU-NEXT:    %4 = stablehlo.convert %0#1 : (tensor<64xi64>) -> tensor<64xi32>
// CPU-NEXT:    %5 = stablehlo.convert %3 : (tensor<64xi64>) -> tensor<64xi32>
// CPU-NEXT:    %6 = stablehlo.convert %0#2 : (tensor<i64>) -> tensor<i32>
// CPU-NEXT:    return %0#0, %4, %5, %6 : tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>
// CPU-NEXT:  }

// CUDA: func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>) {
// CUDA-NEXT:     %c = stablehlo.constant dense<1> : tensor<64xi32>
// CUDA-NEXT:     %0:3 = stablehlo.custom_call @cusolver_getrf_ffi(%arg0) {api_version = 4 : i32, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>)
// CUDA-NEXT:     %1 = stablehlo.subtract %0#1, %c : tensor<64xi32>
// CUDA-NEXT:     %2 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%1) {api_version = 4 : i32, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<64xi32>) -> tensor<64xi32>
// CUDA-NEXT:     %3 = stablehlo.add %c, %2 : tensor<64xi32>
// CUDA-NEXT:     return %0#0, %0#1, %3, %0#2 : tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>
// CUDA-NEXT: }

// TPU: func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>) {
// TPU-NEXT:     %c = stablehlo.constant dense<true> : tensor<i1>
// TPU-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<64xi32>
// TPU-NEXT:     %0:3 = stablehlo.custom_call @LuDecomposition(%arg0) : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>)
// TPU-NEXT:     %1 = stablehlo.add %c_0, %0#1 : tensor<64xi32>
// TPU-NEXT:     %2 = stablehlo.add %c_0, %0#2 : tensor<64xi32>
// TPU-NEXT:     %3 = stablehlo.is_finite %0#0 : (tensor<64x64xf32>) -> tensor<64x64xi1>
// TPU-NEXT:     %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.and across dimensions = [0, 1] : (tensor<64x64xi1>, tensor<i1>) -> tensor<i1>
// TPU-NEXT:     %5 = stablehlo.not %4 : tensor<i1>
// TPU-NEXT:     %6 = stablehlo.convert %5 : (tensor<i1>) -> tensor<i32>
// TPU-NEXT:     return %0#0, %1, %2, %6 : tensor<64x64xf32>, tensor<64xi32>, tensor<64xi32>, tensor<i32>
// TPU-NEXT: }

module {
  // CPU: enzymexla.jit_call @enzymexla_lapack_dgetrf_
  // CPU: func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<i32>) {
  func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<i32>) {
    %0:4 = enzymexla.linalg.lu %arg0 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#3 : tensor<64x64xf64>, tensor<64xi32>, tensor<i32>
  }
}

module {
  // CPU: enzymexla.jit_call @enzymexla_lapack_zgetrf_
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>) {
    %0:4 = enzymexla.linalg.lu %arg0 : (tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#3 : tensor<64x64xcomplex<f64>>, tensor<64xi32>, tensor<i32>
  }
}

module {
  // CPU: enzymexla.jit_call @enzymexla_lapack_cgetrf_
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>) {
    %0:4 = enzymexla.linalg.lu %arg0 : (tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    return %0#0, %0#1, %0#3 : tensor<64x64xcomplex<f32>>, tensor<64xi32>, tensor<i32>
  }
}
