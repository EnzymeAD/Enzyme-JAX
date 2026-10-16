// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x48xf32>) -> tensor<64x48xf32> {
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 <{side = #enzymexla.side<left>}> : (tensor<64x32xf32>, tensor<32xf32>, tensor<64x48xf32>) -> tensor<64x48xf32>
    return %0: tensor<64x48xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sormqr_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.load %arg0 : !llvm.ptr -> i64
// CPU-NEXT:    %1 = llvm.load %arg1 : !llvm.ptr -> i8
// CPU-NEXT:    %2 = llvm.load %arg2 : !llvm.ptr -> i8
// CPU-NEXT:    %3 = llvm.load %arg3 : !llvm.ptr -> i64
// CPU-NEXT:    %4 = llvm.load %arg4 : !llvm.ptr -> i64
// CPU-NEXT:    %5 = llvm.load %arg5 : !llvm.ptr -> i64
// CPU-NEXT:    %6 = llvm.load %arg7 : !llvm.ptr -> i64
// CPU-NEXT:    %7 = llvm.load %arg10 : !llvm.ptr -> i64
// CPU-NEXT:    %8 = llvm.call @enzymexla_lapacke_sormqr_(%0, %1, %2, %3, %4, %5, %arg6, %6, %arg8, %arg9, %7) : (i64, i8, i8, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> i64
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sormqr_(i64, i8, i8, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x48xf32>) -> tensor<64x48xf32> {
// CPU-DAG:    %[[I64_102:[a-z0-9_]+]] = stablehlo.constant dense<102> : tensor<i64>
// CPU-DAG:    %[[I8_76:[a-z0-9_]+]] = stablehlo.constant dense<76> : tensor<i8>
// CPU-DAG:    %[[I8_78:[a-z0-9_]+]] = stablehlo.constant dense<78> : tensor<i8>
// CPU-DAG:    %[[I64_64:[a-z0-9_]+]] = stablehlo.constant dense<64> : tensor<i64>
// CPU-DAG:    %[[I64_48:[a-z0-9_]+]] = stablehlo.constant dense<48> : tensor<i64>
// CPU-DAG:    %[[I64_32:[a-z0-9_]+]] = stablehlo.constant dense<32> : tensor<i64>
// CPU:        %0 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sormqr (%[[I64_102]], %[[I8_76]], %[[I8_78]], %[[I64_64]], %[[I64_48]], %[[I64_32]], %arg0, %[[I64_64]], %arg1, %arg2, %[[I64_64]]) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<64x32xf32>, tensor<i64>, tensor<32xf32>, tensor<64x48xf32>, tensor<i64>) -> tensor<64x48xf32>
// CPU-NEXT:    return %0 : tensor<64x48xf32>
// CPU-NEXT:  }

module {
  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x48xf32>) -> tensor<64x48xf32> {
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 <{side = #enzymexla.side<left>, transpose = #enzymexla.transpose<transpose>}> : (tensor<64x32xf32>, tensor<32xf32>, tensor<64x48xf32>) -> tensor<64x48xf32>
    return %0: tensor<64x48xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sormqr_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.load %arg0 : !llvm.ptr -> i64
// CPU-NEXT:    %1 = llvm.load %arg1 : !llvm.ptr -> i8
// CPU-NEXT:    %2 = llvm.load %arg2 : !llvm.ptr -> i8
// CPU-NEXT:    %3 = llvm.load %arg3 : !llvm.ptr -> i64
// CPU-NEXT:    %4 = llvm.load %arg4 : !llvm.ptr -> i64
// CPU-NEXT:    %5 = llvm.load %arg5 : !llvm.ptr -> i64
// CPU-NEXT:    %6 = llvm.load %arg7 : !llvm.ptr -> i64
// CPU-NEXT:    %7 = llvm.load %arg10 : !llvm.ptr -> i64
// CPU-NEXT:    %8 = llvm.call @enzymexla_lapacke_sormqr_(%0, %1, %2, %3, %4, %5, %arg6, %6, %arg8, %arg9, %7) : (i64, i8, i8, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> i64
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sormqr_(i64, i8, i8, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x48xf32>) -> tensor<64x48xf32> {
// CPU-DAG:    %[[I64_102:[a-z0-9_]+]] = stablehlo.constant dense<102> : tensor<i64>
// CPU-DAG:    %[[I8_76:[a-z0-9_]+]] = stablehlo.constant dense<76> : tensor<i8>
// CPU-DAG:    %[[I8_84:[a-z0-9_]+]] = stablehlo.constant dense<84> : tensor<i8>
// CPU-DAG:    %[[I64_64:[a-z0-9_]+]] = stablehlo.constant dense<64> : tensor<i64>
// CPU-DAG:    %[[I64_48:[a-z0-9_]+]] = stablehlo.constant dense<48> : tensor<i64>
// CPU-DAG:    %[[I64_32:[a-z0-9_]+]] = stablehlo.constant dense<32> : tensor<i64>
// CPU:        %0 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sormqr (%[[I64_102]], %[[I8_76]], %[[I8_84]], %[[I64_64]], %[[I64_48]], %[[I64_32]], %arg0, %[[I64_64]], %arg1, %arg2, %[[I64_64]]) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<64x32xf32>, tensor<i64>, tensor<32xf32>, tensor<64x48xf32>, tensor<i64>) -> tensor<64x48xf32>
// CPU-NEXT:    return %0 : tensor<64x48xf32>
// CPU-NEXT:  }

module {
  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<48x64xf32>) -> tensor<48x64xf32> {
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 <{side = #enzymexla.side<right>, transpose = #enzymexla.transpose<none>}> : (tensor<64x32xf32>, tensor<32xf32>, tensor<48x64xf32>) -> tensor<48x64xf32>
    return %0: tensor<48x64xf32>
  }
}

// CPU:  llvm.func @enzymexla_wrapper_lapacke_sormqr_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.load %arg0 : !llvm.ptr -> i64
// CPU-NEXT:    %1 = llvm.load %arg1 : !llvm.ptr -> i8
// CPU-NEXT:    %2 = llvm.load %arg2 : !llvm.ptr -> i8
// CPU-NEXT:    %3 = llvm.load %arg3 : !llvm.ptr -> i64
// CPU-NEXT:    %4 = llvm.load %arg4 : !llvm.ptr -> i64
// CPU-NEXT:    %5 = llvm.load %arg5 : !llvm.ptr -> i64
// CPU-NEXT:    %6 = llvm.load %arg7 : !llvm.ptr -> i64
// CPU-NEXT:    %7 = llvm.load %arg10 : !llvm.ptr -> i64
// CPU-NEXT:    %8 = llvm.call @enzymexla_lapacke_sormqr_(%0, %1, %2, %3, %4, %5, %arg6, %6, %arg8, %arg9, %7) : (i64, i8, i8, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> i64
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzymexla_lapacke_sormqr_(i64, i8, i8, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> i64
// CPU-NEXT:  func.func @main(%arg0: tensor<64x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<48x64xf32>) -> tensor<48x64xf32> {
// CPU-DAG:    %[[I64_102:[a-z0-9_]+]] = stablehlo.constant dense<102> : tensor<i64>
// CPU-DAG:    %[[I8_82:[a-z0-9_]+]] = stablehlo.constant dense<82> : tensor<i8>
// CPU-DAG:    %[[I8_78:[a-z0-9_]+]] = stablehlo.constant dense<78> : tensor<i8>
// CPU-DAG:    %[[I64_48:[a-z0-9_]+]] = stablehlo.constant dense<48> : tensor<i64>
// CPU-DAG:    %[[I64_64:[a-z0-9_]+]] = stablehlo.constant dense<64> : tensor<i64>
// CPU-DAG:    %[[I64_32:[a-z0-9_]+]] = stablehlo.constant dense<32> : tensor<i64>
// CPU:        %0 = enzymexla.jit_call @enzymexla_wrapper_lapacke_sormqr (%[[I64_102]], %[[I8_82]], %[[I8_78]], %[[I64_48]], %[[I64_64]], %[[I64_32]], %arg0, %[[I64_64]], %arg1, %arg2, %[[I64_48]]) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<64x32xf32>, tensor<i64>, tensor<32xf32>, tensor<48x64xf32>, tensor<i64>) -> tensor<48x64xf32>
// CPU-NEXT:    return %0 : tensor<48x64xf32>
// CPU-NEXT:  }

module {
  func.func @main(%arg0: tensor<64x32xf64>, %arg1: tensor<32xf64>, %arg2: tensor<64x48xf64>) -> tensor<64x48xf64> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_dormqr (
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 <{side = #enzymexla.side<left>}> : (tensor<64x32xf64>, tensor<32xf64>, tensor<64x48xf64>) -> tensor<64x48xf64>
    return %0: tensor<64x48xf64>
  }
}

module {
  func.func @main(%arg0: tensor<64x32xcomplex<f32>>, %arg1: tensor<32xcomplex<f32>>, %arg2: tensor<64x48xcomplex<f32>>) -> tensor<64x48xcomplex<f32>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_cunmqr_ (
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 <{side = #enzymexla.side<left>}> : (tensor<64x32xcomplex<f32>>, tensor<32xcomplex<f32>>, tensor<64x48xcomplex<f32>>) -> tensor<64x48xcomplex<f32>>
    return %0: tensor<64x48xcomplex<f32>>
  }
}

module {
  func.func @main(%arg0: tensor<64x32xcomplex<f64>>, %arg1: tensor<32xcomplex<f64>>, %arg2: tensor<64x48xcomplex<f64>>) -> tensor<64x48xcomplex<f64>> {
    // CPU: enzymexla.jit_call @enzymexla_wrapper_lapacke_zunmqr_ (
    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 <{side = #enzymexla.side<left>}> : (tensor<64x32xcomplex<f64>>, tensor<32xcomplex<f64>>, tensor<64x48xcomplex<f64>>) -> tensor<64x48xcomplex<f64>>
    return %0: tensor<64x48xcomplex<f64>>
  }
}
