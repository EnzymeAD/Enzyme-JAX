// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %0 = stablehlo.slice %arg0 [1:2, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
    %1 = stablehlo.slice %arg0 [2:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1518x3056xf64>
    %2 = stablehlo.slice %arg1 [1:1519, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1518x3056xf64>
    %3 = stablehlo.slice %arg1 [1519:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
    %4 = stablehlo.subtract %1, %2 : tensor<1518x3056xf64>
    %5 = stablehlo.negate %3 : tensor<1x3056xf64>
    %6 = stablehlo.concatenate %0, %4, %5, dim = 0 : (tensor<1x3056xf64>, tensor<1518x3056xf64>, tensor<1x3056xf64>) -> tensor<1520x3056xf64>
    return %6 : tensor<1520x3056xf64>
}

// CHECK: func.func @main1(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [1:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1519x3056xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     %2 = stablehlo.slice %arg1 [1:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:     %3 = stablehlo.pad %2, %cst, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1519x3056xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     %4 = stablehlo.subtract %1, %3 : tensor<1520x3056xf64>
// CHECK-NEXT:     return %4 : tensor<1520x3056xf64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>, %arg2: tensor<1536xf64>) -> tensor<1520x3056xf64> {
    %0 = stablehlo.slice %arg0 [1:2, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
    %m1 = stablehlo.slice %arg2 [9:10] : (tensor<1536xf64>) -> tensor<1xf64>
    %m1_b = stablehlo.broadcast_in_dim %m1, dims = [0] : (tensor<1xf64>) -> tensor<1x3056xf64>
    %l = stablehlo.multiply %m1_b, %0 : tensor<1x3056xf64>
    %m2 = stablehlo.slice %arg2 [10:1528] : (tensor<1536xf64>) -> tensor<1518xf64>
    %m2_b = stablehlo.broadcast_in_dim %m2, dims = [0] : (tensor<1518xf64>) -> tensor<1518x3056xf64>
    %1 = stablehlo.slice %arg0 [2:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1518x3056xf64>
    %m = stablehlo.multiply %m2_b, %1 : tensor<1518x3056xf64>
    %2 = stablehlo.slice %arg1 [1:1519, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1518x3056xf64>
    %3 = stablehlo.slice %arg1 [1519:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
    %4 = stablehlo.subtract %m, %2 : tensor<1518x3056xf64>
    %5 = stablehlo.negate %3 : tensor<1x3056xf64>
    %6 = stablehlo.concatenate %l, %4, %5, dim = 0 : (tensor<1x3056xf64>, tensor<1518x3056xf64>, tensor<1x3056xf64>) -> tensor<1520x3056xf64>
    return %6 : tensor<1520x3056xf64>
}

// CHECK: func.func @main2(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>, %arg2: tensor<1536xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg2 [9:10] : (tensor<1536xf64>) -> tensor<1xf64>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf64>) -> tensor<1x3056xf64>
// CHECK-NEXT:     %2 = stablehlo.slice %arg2 [10:1528] : (tensor<1536xf64>) -> tensor<1518xf64>
// CHECK-NEXT:     %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<1518xf64>) -> tensor<1518x3056xf64>
// CHECK-NEXT:     %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1x3056xf64>, tensor<1518x3056xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:     %5 = stablehlo.slice %arg0 [1:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:     %6 = stablehlo.multiply %4, %5 : tensor<1519x3056xf64>
// CHECK-NEXT:     %7 = stablehlo.pad %6, %cst, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1519x3056xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     %8 = stablehlo.slice %arg1 [1:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1519x3056xf64>
// CHECK-NEXT:     %9 = stablehlo.pad %8, %cst, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1519x3056xf64>, tensor<f64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:     %10 = stablehlo.subtract %7, %9 : tensor<1520x3056xf64>
// CHECK-NEXT:     return %10 : tensor<1520x3056xf64>
// CHECK-NEXT: }
