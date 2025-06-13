// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_if" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @transpose_if(%arg0: tensor<32x32x1x32xf32>, %arg1: tensor<i1>, %arg2: tensor<32x1x32xf32>, %arg3: tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32> {
  %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<32x32x1x32xf32>
  %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<32x32x1x32xf32>
  %1 = "stablehlo.if"(%arg1) ({
    stablehlo.return %arg0 : tensor<32x32x1x32xf32>
  }, {
    %3 = chlo.is_inf %arg2 : tensor<32x1x32xf32> -> tensor<32x1x32xi1>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2, 3] : (tensor<32x1x32xi1>) -> tensor<32x32x1x32xi1>
    %5 = chlo.is_inf %arg3 : tensor<32x32x1x32xf32> -> tensor<32x32x1x32xi1>
    %6 = stablehlo.select %5, %cst_11, %cst_12 : tensor<32x32x1x32xi1>, tensor<32x32x1x32xf32>
    %7 = stablehlo.select %4, %6, %arg0 : tensor<32x32x1x32xi1>, tensor<32x32x1x32xf32>
    stablehlo.return %7 : tensor<32x32x1x32xf32>
  }) : (tensor<i1>) -> tensor<32x32x1x32xf32>
  %2 = stablehlo.transpose %1, dims = [3, 2, 1, 0] : (tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32>
  return %2 : tensor<32x1x32x32xf32>
}

// CHECK: func.func @transpose_if(%arg0: tensor<32x32x1x32xf32>, %arg1: tensor<i1>, %arg2: tensor<32x1x32xf32>, %arg3: tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<1.000000e+00> : tensor<32x32x1x32xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<32x32x1x32xf32>
// CHECK-NEXT:   %0 = "stablehlo.if"(%arg1) ({
// CHECK-NEXT:     %1 = stablehlo.transpose %arg0, dims = [3, 2, 1, 0] : (tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32>
// CHECK-NEXT:     stablehlo.return %1 : tensor<32x1x32x32xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:     %1 = chlo.is_inf %arg2 : tensor<32x1x32xf32> -> tensor<32x1x32xi1>
// CHECK-NEXT:     %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2, 3] : (tensor<32x1x32xi1>) -> tensor<32x32x1x32xi1>
// CHECK-NEXT:     %3 = chlo.is_inf %arg3 : tensor<32x32x1x32xf32> -> tensor<32x32x1x32xi1>
// CHECK-NEXT:     %4 = stablehlo.select %3, %cst, %cst_0 : tensor<32x32x1x32xi1>, tensor<32x32x1x32xf32>
// CHECK-NEXT:     %5 = stablehlo.select %2, %4, %arg0 : tensor<32x32x1x32xi1>, tensor<32x32x1x32xf32>
// CHECK-NEXT:     %6 = stablehlo.transpose %5, dims = [3, 2, 1, 0] : (tensor<32x32x1x32xf32>) -> tensor<32x1x32x32xf32>
// CHECK-NEXT:     stablehlo.return %6 : tensor<32x1x32x32xf32>
// CHECK-NEXT:   }) : (tensor<i1>) -> tensor<32x1x32x32xf32>
// CHECK-NEXT:   return %0 : tensor<32x1x32x32xf32>
// CHECK-NEXT: }
