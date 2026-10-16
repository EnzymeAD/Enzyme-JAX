// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>) {
    %0:3 = enzymexla.lapack.geqrt %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sgeqrt(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.load %arg0 : !llvm.ptr -> i64
// CPU-NEXT:    %1 = llvm.load %arg1 : !llvm.ptr -> i64
// CPU-NEXT:    %2 = llvm.load %arg2 : !llvm.ptr -> i64
// CPU-NEXT:    %3 = llvm.load %arg3 : !llvm.ptr -> i64
// CPU-NEXT:    %4 = llvm.load %arg5 : !llvm.ptr -> i64
// CPU-NEXT:    %5 = llvm.load %arg7 : !llvm.ptr -> i64
// CPU-NEXT:    %6 = llvm.call @enzymexla_lapacke_sgeqrt(%0, %1, %2, %3, %arg4, %4, %arg6, %5) : (i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:    llvm.store %6, %arg8 : i64, !llvm.ptr
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sgeqrt(i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>) {
// CPU-DAG:    %[[C:[a-z0-9_]+]] = stablehlo.constant dense<-1> : tensor<i64>
// CPU-DAG:    %[[CST:[a-z0-9_]+]] = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf32>
// CPU-DAG:    %[[I64_102:[a-z0-9_]+]] = stablehlo.constant dense<102> : tensor<i64>
// CPU-DAG:    %[[I64_64:[a-z0-9_]+]] = stablehlo.constant dense<64> : tensor<i64>
// CPU:        %0:3 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sgeqrt (%[[I64_102]], %[[I64_64]], %[[I64_64]], %[[I64_64]], %arg0, %[[I64_64]], %[[CST]], %[[I64_64]], %[[C]]) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 6, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 8, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<64x64xf32>, tensor<i64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>)
// CPU-NEXT:    return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<i64>
// CPU-NEXT:  }

module {
  // CPU: func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64x64xf64>, tensor<i64>) {
  func.func @main(%arg0: tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64x64xf64>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dgeqrt (
    %0:3 = enzymexla.lapack.geqrt %arg0 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64x64xf64>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xf64>, tensor<64x64xf64>, tensor<i64>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64x64xcomplex<f32>>, tensor<i64>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64x64xcomplex<f32>>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cgeqrt (
    %0:3 = enzymexla.lapack.geqrt %arg0 : (tensor<64x64xcomplex<f32>>) -> (tensor<64x64xcomplex<f32>>, tensor<64x64xcomplex<f32>>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xcomplex<f32>>, tensor<64x64xcomplex<f32>>, tensor<i64>
  }
}

module {
  // CPU: func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64x64xcomplex<f64>>, tensor<i64>) {
  func.func @main(%arg0: tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64x64xcomplex<f64>>, tensor<i64>) {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zgeqrt (
    %0:3 = enzymexla.lapack.geqrt %arg0 : (tensor<64x64xcomplex<f64>>) -> (tensor<64x64xcomplex<f64>>, tensor<64x64xcomplex<f64>>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<64x64xcomplex<f64>>, tensor<64x64xcomplex<f64>>, tensor<i64>
  }
}
