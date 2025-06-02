// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @divide1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.divide %arg1, %arg2 : tensor<2x2xf32>
    %1 = stablehlo.divide %arg0, %0 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// CHECK: func.func @divide1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.multiply %arg0, %arg2 : tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.divide %0, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:   return %1 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @divide2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<2x2xf32>
    %1 = stablehlo.divide %0, %arg2 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// CHECK: func.func @divide2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.multiply %arg1, %arg2 : tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.divide %arg0, %0 : tensor<2x2xf32>
// CHECK-NEXT:   return %1 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @divide3(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<2x2xf32>
    %1 = stablehlo.divide %arg2, %arg3 : tensor<2x2xf32>
    %2 = stablehlo.divide %0, %1 : tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
}

// CHECK: func.func @divide3(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-DAG: %[[MUL1:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<2x2xf32>
// CHECK-DAG: %[[MUL2:.*]] = stablehlo.multiply %arg0, %arg3 : tensor<2x2xf32>
// CHECK:     %[[DIV:.*]] = stablehlo.divide %[[MUL2]], %[[MUL1]] : tensor<2x2xf32>
// CHECK-NEXT:     return %[[DIV]] : tensor<2x2xf32>
// CHECK-NEXT:   }
