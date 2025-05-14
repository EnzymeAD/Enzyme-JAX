// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
    %0:3 = enzymexla.linalg.lu %arg0 : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>)
    return %0#0, %0#1, %0#2 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>
  }
}

// CPU:  llvm.func @enzymexla_lapack_sgetrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU-NEXT:  llvm.func @enzymexla_lapack_sgetrf_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %2 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %3 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    llvm.store %0, %2 : i64, !llvm.ptr
// CPU-NEXT:    llvm.store %0, %3 : i64, !llvm.ptr
// CPU-NEXT:    llvm.call @enzymexla_lapack_sgetrf_(%2, %3, %arg0, %2, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
// CPU-NEXT:    %c = stablehlo.constant dense<1> : tensor<i32>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<-1> : tensor<64xi64>
// CPU-NEXT:    %c_2 = stablehlo.constant dense<12> : tensor<i32>
// CPU-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<i32>
// CPU-NEXT:    %c_4 = stablehlo.constant dense<-1> : tensor<12xi64>
// CPU-NEXT:    %c_5 = stablehlo.constant dense<-1> : tensor<12x64xi64>
// CPU-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<4x3x64x64xf32>) -> tensor<12x64x64xf32>
// CPU-NEXT:    %1:4 = stablehlo.while(%iterArg = %c_3, %iterArg_6 = %0, %iterArg_7 = %c_5, %iterArg_8 = %c_4) : tensor<i32>, tensor<12x64x64xf32>, tensor<12x64xi64>, tensor<12xi64>
// CPU-NEXT:     cond {
// CPU-NEXT:      %7 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CPU-NEXT:      stablehlo.return %7 : tensor<i1>
// CPU-NEXT:    } do {
// CPU-NEXT:      %7 = stablehlo.dynamic_slice %iterArg_6, %iterArg, %c_3, %c_3, sizes = [1, 64, 64] : (tensor<12x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x64x64xf32>
// CPU-NEXT:      %8 = stablehlo.reshape %7 : (tensor<1x64x64xf32>) -> tensor<64x64xf32>
// CPU-NEXT:      %9:3 = enzymexla.jit_call @enzymexla_lapack_sgetrf_wrapper_0 (%8, %c_1, %c_0) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], xla_side_effect_free} : (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>) -> (tensor<64x64xf32>, tensor<64xi64>, tensor<i64>)
// CPU-NEXT:      %10 = stablehlo.reshape %9#0 : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
// CPU-NEXT:      %11 = stablehlo.dynamic_update_slice %iterArg_6, %10, %iterArg, %c_3, %c_3 : (tensor<12x64x64xf32>, tensor<1x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<12x64x64xf32>
// CPU-NEXT:      %12 = stablehlo.reshape %9#1 : (tensor<64xi64>) -> tensor<1x64xi64>
// CPU-NEXT:      %13 = stablehlo.dynamic_update_slice %iterArg_7, %12, %iterArg, %c_3 : (tensor<12x64xi64>, tensor<1x64xi64>, tensor<i32>, tensor<i32>) -> tensor<12x64xi64>
// CPU-NEXT:      %14 = stablehlo.reshape %9#2 : (tensor<i64>) -> tensor<1xi64>
// CPU-NEXT:      %15 = stablehlo.dynamic_update_slice %iterArg_8, %14, %iterArg : (tensor<12xi64>, tensor<1xi64>, tensor<i32>) -> tensor<12xi64>
// CPU-NEXT:      %16 = stablehlo.add %iterArg, %c : tensor<i32>
// CPU-NEXT:      stablehlo.return %16, %11, %13, %15 : tensor<i32>, tensor<12x64x64xf32>, tensor<12x64xi64>, tensor<12xi64>
// CPU-NEXT:    }
// CPU-NEXT:    %2 = stablehlo.reshape %1#1 : (tensor<12x64x64xf32>) -> tensor<4x3x64x64xf32>
// CPU-NEXT:    %3 = stablehlo.convert %1#2 : (tensor<12x64xi64>) -> tensor<12x64xi32>
// CPU-NEXT:    %4 = stablehlo.reshape %3 : (tensor<12x64xi32>) -> tensor<4x3x64xi32>
// CPU-NEXT:    %5 = stablehlo.convert %1#3 : (tensor<12xi64>) -> tensor<12xi32>
// CPU-NEXT:    %6 = stablehlo.reshape %5 : (tensor<12xi32>) -> tensor<4x3xi32>
// CPU-NEXT:    return %2, %4, %6 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>
// CPU-NEXT:  }


// CUDA: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
// CUDA-NEXT:     %0:3 = stablehlo.custom_call @cusolver_getrf_ffi(%arg0) {operand_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>)
// CUDA-NEXT:     return %0#0, %0#1, %0#2 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>
// CUDA-NEXT: }

// TPU: func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>) {
// TPU-NEXT:     %c = stablehlo.constant dense<true> : tensor<i1>
// TPU-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<4x3x64xi32>
// TPU-NEXT:     %0:3 = stablehlo.custom_call @LUFactorization(%arg0) : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3x64xi32>)
// TPU-NEXT:     %1 = stablehlo.add %c_0, %0#1 : tensor<4x3x64xi32>
// TPU-NEXT:     %2 = stablehlo.is_finite %0#0 : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xi1>
// TPU-NEXT:     %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.and across dimensions = [2, 3] : (tensor<4x3x64x64xi1>, tensor<i1>) -> tensor<4x3xi1>
// TPU-NEXT:     %4 = stablehlo.not %3 : tensor<4x3xi1>
// TPU-NEXT:     %5 = stablehlo.convert %4 : (tensor<4x3xi1>) -> tensor<4x3xi32>
// TPU-NEXT:     return %0#0, %1, %5 : tensor<4x3x64x64xf32>, tensor<4x3x64xi32>, tensor<4x3xi32>
// TPU-NEXT: }
