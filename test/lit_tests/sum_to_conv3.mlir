// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=sum_to_conv;convert_simplify;reshape_op_canon;noop_slice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module @reactant_problem... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3>]
  func.func @main(%arg0: tensor<3x32x16xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<3x32x16xf64>) -> tensor<1x32x16xf64> {
    %0 = stablehlo.slice %arg1 [0:1, 0:32, 0:16] : (tensor<3x32x16xf64>) -> tensor<1x32x16xf64>
    %1 = stablehlo.slice %arg1 [1:2, 0:32, 0:16] : (tensor<3x32x16xf64>) -> tensor<1x32x16xf64>
    %2 = stablehlo.slice %arg0 [1:2, 0:32, 0:16] : (tensor<3x32x16xf64>) -> tensor<1x32x16xf64>
    %3 = stablehlo.slice %arg1 [2:3, 0:32, 0:16] : (tensor<3x32x16xf64>) -> tensor<1x32x16xf64>
    %4 = stablehlo.add %2, %0 : tensor<1x32x16xf64> // arg0[1] + arg1[0]
    %5 = stablehlo.add %4, %1 : tensor<1x32x16xf64> // (arg0[1] + arg1[0]) + arg1[1]
    %6 = stablehlo.add %1, %3 : tensor<1x32x16xf64> // arg1[1] + arg1[2]
    %7 = stablehlo.add %6, %5 : tensor<1x32x16xf64> // (arg1[1] + arg1[2]) + ((arg0[1] + arg1[0]) + arg1[1]) == (arg0[1]) + (arg1[0] + 2 * arg1[1] + arg1[2])
    return %7 : tensor<1x32x16xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3x32x16xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<3x32x16xf64>) -> tensor<1x32x16xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[[1.000000e+00]], [[2.000000e+00]], [[1.000000e+00]]]> : tensor<3x1x1xf64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [1:2, 0:32, 0:16] : (tensor<3x32x16xf64>) -> tensor<1x32x16xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg1 : (tensor<3x32x16xf64>) -> tensor<3x512x1xf64>
// CHECK-NEXT:    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [0, b, f]x[0, i, o]->[0, b, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x512x1xf64>, tensor<3x1x1xf64>) -> tensor<1x512x1xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x512x1xf64>) -> tensor<1x32x16xf64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %0 : tensor<1x32x16xf64>
// CHECK-NEXT:    return %4 : tensor<1x32x16xf64>
// CHECK-NEXT:  }