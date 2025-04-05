// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=sum_to_conv;convert_simplify;reshape_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module @reactant_simple_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3>]
  func.func @main(%arg0: tensor<1092xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<1095xf64>) -> tensor<1092xf64> {
    %0 = stablehlo.slice %arg1 [0:1092] : (tensor<1095xf64>) -> tensor<1092xf64>
    %1 = stablehlo.slice %arg1 [1:1093] : (tensor<1095xf64>) -> tensor<1092xf64>
    %2 = stablehlo.slice %arg1 [2:1094] : (tensor<1095xf64>) -> tensor<1092xf64>
    %3 = stablehlo.add %0, %1 : tensor<1092xf64>
    %4 = stablehlo.add %3, %2 : tensor<1092xf64>
    return %4 : tensor<1092xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1092xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<1095xf64>) -> tensor<1092xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x1x1xf64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1094] : (tensor<1095xf64>) -> tensor<1094xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1094xf64>) -> tensor<1x1x1094xf64>
// CHECK-NEXT:    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0]x[0, i, o]->[b, f, 0], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1094xf64>, tensor<3x1x1xf64>) -> tensor<1x1x1092xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1x1092xf64>) -> tensor<1092xf64>
// CHECK-NEXT:    return %3 : tensor<1092xf64>
// CHECK-NEXT:  }