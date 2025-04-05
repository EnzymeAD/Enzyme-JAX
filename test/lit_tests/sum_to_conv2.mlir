// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=sum_to_conv;convert_simplify;reshape_op_canon;noop_slice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module @reactant_simple_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%200: tensor<128x1007x1008xf64>) -> tensor<127x1007x1008xf64> {
      %440 = stablehlo.slice %200 [1:128, 0:1007, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<127x1007x1008xf64>
      %441 = stablehlo.slice %200 [0:127, 0:1007, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<127x1007x1008xf64>
      %442 = stablehlo.subtract %440, %441 : tensor<127x1007x1008xf64>
    return %442 : tensor<127x1007x1008xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<128x1007x1008xf64>) -> tensor<127x1007x1008xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[[-1.000000e+00]], [[1.000000e+00]]]> : tensor<2x1x1xf64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<128x1007x1008xf64>) -> tensor<128x1015056x1xf64>
// CHECK-NEXT:    %1 = stablehlo.convolution(%0, %cst) dim_numbers = [0, b, f]x[0, i, o]->[0, b, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x1015056x1xf64>, tensor<2x1x1xf64>) -> tensor<127x1015056x1xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<127x1015056x1xf64>) -> tensor<127x1007x1008xf64>
// CHECK-NEXT:    return %2 : tensor<127x1007x1008xf64>
// CHECK-NEXT:  }