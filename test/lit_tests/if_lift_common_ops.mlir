// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=if_op_lift_common_ops" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @transpose_if(%arg0: tensor<32x32x1x32xf32>, %arg1: tensor<i1>, %arg2: tensor<32x1x32xf32>, %arg3: tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x1x32x32xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<32x1x32x32xf32>
    %0 = chlo.is_inf %arg3 : tensor<32x32x1x32xf32> -> tensor<32x32x1x32xi1>
    %1 = stablehlo.transpose %0, dims = [3, 2, 1, 0] : (tensor<32x32x1x32xi1>) -> tensor<32x1x32x32xi1>
    %2 = "stablehlo.if"(%arg1) ({
      %3 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32>
      stablehlo.return %3 : tensor<32x1x32x32xf32>
    }, {
      %3 = chlo.is_inf %arg2 : tensor<32x1x32xf32> -> tensor<32x1x32xi1>
      %4 = stablehlo.broadcast_in_dim %3, dims = [2, 1, 0] : (tensor<32x1x32xi1>) -> tensor<32x1x32x32xi1>
      %5 = stablehlo.select %1, %cst_0, %cst : tensor<32x1x32x32xi1>, tensor<32x1x32x32xf32>
      %6 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32>
      %7 = stablehlo.select %4, %5, %6 : tensor<32x1x32x32xi1>, tensor<32x1x32x32xf32>
      stablehlo.return %7 : tensor<32x1x32x32xf32>
    }) : (tensor<i1>) -> tensor<32x1x32x32xf32>
    return %2 : tensor<32x1x32x32xf32>
  }
}

// CHECK: func.func @transpose_if(%arg0: tensor<32x32x1x32xf32>, %arg1: tensor<i1>, %arg2: tensor<32x1x32xf32>, %arg3: tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<32x1x32x32xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<32x1x32x32xf32>
// CHECK-NEXT:     %0 = chlo.is_inf %arg3 : tensor<32x32x1x32xf32> -> tensor<32x32x1x32xi1>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [3, 2, 1, 0] : (tensor<32x32x1x32xi1>) -> tensor<32x1x32x32xi1>
// CHECK-NEXT:     %2 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32>
// CHECK-NEXT:     %3 = "stablehlo.if"(%arg1) ({
// CHECK-NEXT:       stablehlo.return %2 : tensor<32x1x32x32xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %4 = chlo.is_inf %arg2 : tensor<32x1x32xf32> -> tensor<32x1x32xi1>
// CHECK-NEXT:       %5 = stablehlo.broadcast_in_dim %4, dims = [2, 1, 0] : (tensor<32x1x32xi1>) -> tensor<32x1x32x32xi1>
// CHECK-NEXT:       %6 = stablehlo.select %1, %cst_0, %cst : tensor<32x1x32x32xi1>, tensor<32x1x32x32xf32>
// CHECK-NEXT:       %7 = stablehlo.select %5, %6, %2 : tensor<32x1x32x32xi1>, tensor<32x1x32x32xf32>
// CHECK-NEXT:       stablehlo.return %7 : tensor<32x1x32x32xf32>
// CHECK-NEXT:     }) : (tensor<i1>) -> tensor<32x1x32x32xf32>
// CHECK-NEXT:     return %3 : tensor<32x1x32x32xf32>
// CHECK-NEXT: }
