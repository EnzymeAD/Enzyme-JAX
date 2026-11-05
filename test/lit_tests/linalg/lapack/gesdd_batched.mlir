// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt,drop-unsupported-attributes)" %s | FileCheck %s --check-prefix=CPU

func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
  %0:4 = enzymexla.lapack.gesdd %arg0 : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
}

// CPU: llvm.func private @enzymexla_wrapper_lapack_sgesdd_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
// CPU-NEXT:     %0 = llvm.mlir.constant(8 : i64) : i64
// CPU-NEXT:     %1 = llvm.mlir.constant(83 : i8) : i8
// CPU-NEXT:     %2 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %3 = llvm.mlir.constant(-1 : i64) : i64
// CPU-NEXT:     %4 = llvm.alloca %2 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %3, %4 : i64, !llvm.ptr
// CPU-NEXT:     %5 = llvm.alloca %2 x f32 : (i64) -> !llvm.ptr
// CPU-NEXT:     %6 = llvm.alloca %2 x i8 : (i64) -> !llvm.ptr
// CPU-NEXT:     %7 = llvm.load %arg0 : !llvm.ptr -> i64
// CPU-NEXT:     %8 = llvm.load %arg1 : !llvm.ptr -> i64
// CPU-NEXT:     %9 = arith.minsi %7, %8 : i64
// CPU-NEXT:     %10 = llvm.mul %9, %0 : i64
// CPU-NEXT:     %11 = llvm.alloca %10 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %1, %6 : i8, !llvm.ptr
// CPU-NEXT:     llvm.call @enzymexla_lapack_sgesdd_(%6, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %5, %4, %11, %arg9, %2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
// CPU-NEXT:     %12 = llvm.load %5 : !llvm.ptr -> f32
// CPU-NEXT:     %13 = llvm.fptosi %12 : f32 to i64
// CPU-NEXT:     %14 = llvm.alloca %13 x f32 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %13, %4 : i64, !llvm.ptr
// CPU-NEXT:     llvm.call @enzymexla_lapack_sgesdd_(%6, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %14, %4, %11, %arg9, %2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }
// CPU-NEXT: llvm.func private @enzymexla_wrapper_lapack_sgesdd__batched(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
// CPU-NEXT:   affine.for %arg10 = 0 to 12 {
// CPU-NEXT:     %0 = arith.index_cast %arg10 : index to i64
// CPU-NEXT:     %1 = llvm.getelementptr inbounds %arg2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16384 x i8>
// CPU-NEXT:     %2 = llvm.getelementptr inbounds %arg4[%0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x i8>
// CPU-NEXT:     %3 = llvm.getelementptr inbounds %arg5[%0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16384 x i8>
// CPU-NEXT:     %4 = llvm.getelementptr inbounds %arg7[%0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16384 x i8>
// CPU-NEXT:     %5 = llvm.getelementptr inbounds %arg9[%0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
// CPU-NEXT:     llvm.call @enzymexla_wrapper_lapack_sgesdd_(%arg0, %arg1, %1, %arg3, %2, %3, %arg6, %4, %arg8, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:   }
// CPU-NEXT:   llvm.return
// CPU-NEXT: }
// CPU-NEXT:   llvm.func @enzymexla_lapack_sgesdd_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
// CPU-NEXT:   func.func @main(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// CPU-NEXT:     %0:4 = call @batched_shlo_enzymexla_wrapper_lapack_sgesdd__wrapper_0(%arg0) : (tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>)
// CPU-NEXT:     return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CPU-NEXT:   }
// CPU-NEXT: func.func private @batched_shlo_enzymexla_wrapper_lapack_sgesdd__wrapper_0(%arg0: tensor<4x3x64x64xf32>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>) {
// CPU-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x3x64x64xf32>
// CPU-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x3x64xf32>
// CPU-NEXT:   %c = stablehlo.constant dense<0> : tensor<4x3xi64>
// CPU-NEXT:   %c_1 = stablehlo.constant dense<64> : tensor<i64>
// CPU-NEXT:   %0:5 = enzymexla.jit_call @enzymexla_wrapper_lapack_sgesdd__batched (%c_1, %c_1, %arg0, %c_1, %cst_0, %cst, %c_1, %cst, %c_1, %c) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<> : tensor<0xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<> : tensor<0xindex>, dense<[2, 3, 1, 0]> : tensor<4xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 9, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 2, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<4x3x64x64xf32>, tensor<i64>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<i64>, tensor<4x3x64x64xf32>, tensor<i64>, tensor<4x3xi64>) -> (tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>, tensor<4x3x64x64xf32>)
// CPU-NEXT:   return %0#0, %0#1, %0#2, %0#3 : tensor<4x3x64x64xf32>, tensor<4x3x64xf32>, tensor<4x3x64x64xf32>, tensor<4x3xi64>
// CPU-NEXT: }
