// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3>]
  func.func @"problem_kernel!"(%arg0: tensor<9x32x16xf64>, %arg1: tensor<9x32x16xf64>) -> (tensor<9x32x16xf64>, tensor<9x32x16xf64>) {
    %0 = enzymexla.jit_call @"##call__Z14gpu__integral_16CompilerMetadataI10StaticSizeI11_16__32__8_E12DynamicCheckvv7NDRangeILi3ES0_I9_1__2__8_ES0_I11_16__16__1_EvvEE13CuTracedArrayI7Float64Li3ELi1E11_16__32__9_E5Int64SA_#358$par13" (%arg0, %arg1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<9x32x16xf64>, tensor<9x32x16xf64>) -> tensor<9x32x16xf64>
    return %0, %arg1 : tensor<9x32x16xf64>, tensor<9x32x16xf64>
  }
  func.func private @"##call__Z14gpu__integral_16CompilerMetadataI10StaticSizeI11_16__32__8_E12DynamicCheckvv7NDRangeILi3ES0_I9_1__2__8_ES0_I11_16__16__1_EvvEE13CuTracedArrayI7Float64Li3ELi1E11_16__32__9_E5Int64SA_#358$par13"(%arg0: memref<9x32x16xf64, 1>, %arg1: memref<9x32x16xf64, 1>) {
    affine.parallel (%arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0) to (32, 2, 2, 2, 16) {
      %0 = affine.load %arg1[7, %arg2, %arg6] : memref<9x32x16xf64, 1>
      %1 = affine.load %arg1[8, %arg2, %arg6] : memref<9x32x16xf64, 1>
      %2 = arith.addf %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %2, %arg0[7, %arg2, %arg6] : memref<9x32x16xf64, 1>
      %3 = affine.load %arg0[7, %arg2, %arg6] : memref<9x32x16xf64, 1>
      %4 = affine.for %arg7 = 0 to 7 iter_args(%arg8 = %3) -> (f64) {
        %5 = affine.load %arg1[-%arg7 + 6, %arg2, %arg6] : memref<9x32x16xf64, 1>
        %6 = affine.load %arg1[-%arg7 + 7, %arg2, %arg6] : memref<9x32x16xf64, 1>
        %7 = arith.addf %arg8, %5 {fastmathFlags = #llvm.fastmath<none>} : f64
        %8 = arith.addf %7, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %8, %arg0[-%arg7 + 6, %arg2, %arg6] : memref<9x32x16xf64, 1>
        affine.yield %8 : f64
      }
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z14gpu__integral_16CompilerMetadataI10StaticSizeI11_16__32__8_E12DynamicCheckvv7NDRangeILi3ES0_I9_1__2__8_ES0_I11_16__16__1_EvvEE13CuTracedArrayI7Float64Li3ELi1E11_16__32__9_E5Int64SA_#358$par13_raised"(%arg0: tensor<9x32x16xf64>, %arg1: tensor<9x32x16xf64>) -> (tensor<9x32x16xf64>, tensor<9x32x16xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [7:8, 0:32, 0:16] : (tensor<9x32x16xf64>) -> tensor<1x32x16xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x32x16xf64>) -> tensor<32x16xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg1 [8:9, 0:32, 0:16] : (tensor<9x32x16xf64>) -> tensor<1x32x16xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x32x16xf64>) -> tensor<32x16xf64>
// CHECK-NEXT:    %4 = arith.addf %1, %3 {fastmathFlags = #llvm.fastmath<none>} : tensor<32x16xf64>
// CHECK-NEXT:    %5 = stablehlo.reshape %4 : (tensor<32x16xf64>) -> tensor<1x32x16xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %arg0 [8:9, 0:32, 0:16] : (tensor<9x32x16xf64>) -> tensor<1x32x16xf64>
// CHECK-NEXT:    %7 = stablehlo.slice %arg1 [0:7, 0:32, 0:16] : (tensor<9x32x16xf64>) -> tensor<7x32x16xf64>
// CHECK-NEXT:    %8 = stablehlo.reverse %7, dims = [0] : tensor<7x32x16xf64>
// CHECK-NEXT:    %9 = stablehlo.slice %arg1 [1:8, 0:32, 0:16] : (tensor<9x32x16xf64>) -> tensor<7x32x16xf64>
// CHECK-NEXT:    %10 = stablehlo.reverse %9, dims = [0] : tensor<7x32x16xf64>
// CHECK-NEXT:    %11 = arith.addf %10, %8 : tensor<7x32x16xf64>
// CHECK-NEXT:    %12 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<32x16xf64>) -> tensor<7x32x16xf64>
// CHECK-NEXT{LITERAL}:    %13 = "stablehlo.reduce_window"(%11, %cst) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[6, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 7, 1, 1>, window_strides = array<i64: 1, 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %17 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %17 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<7x32x16xf64>, tensor<f64>) -> tensor<7x32x16xf64>
// CHECK-NEXT:    %14 = stablehlo.add %13, %12 : tensor<7x32x16xf64>
// CHECK-NEXT:    %15 = stablehlo.reverse %14, dims = [0] : tensor<7x32x16xf64>
// CHECK-NEXT:    %16 = stablehlo.concatenate %15, %5, %6, dim = 0 : (tensor<7x32x16xf64>, tensor<1x32x16xf64>, tensor<1x32x16xf64>) -> tensor<9x32x16xf64>
// CHECK-NEXT:    return %16, %arg1 : tensor<9x32x16xf64>, tensor<9x32x16xf64>
// CHECK-NEXT:  }
