// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @cseadd(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
    %1 = stablehlo.add %arg1, %arg0 : tensor<2x2xf32>
    %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
}

// CHECK: func.func @cseadd(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.add %0, %0 : tensor<2x2xf32>
// CHECK-NEXT:   return %1 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @csemul(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x2xf32>
    %1 = stablehlo.multiply %arg1, %arg0 : tensor<2x2xf32>
    %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
}

// CHECK: func.func @csemul(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.add %0, %0 : tensor<2x2xf32>
// CHECK-NEXT:   return %1 : tensor<2x2xf32>
// CHECK-NEXT: }

// shouldn't apply to division
func.func @csediv(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<2x2xf32>
    %1 = stablehlo.divide %arg1, %arg0 : tensor<2x2xf32>
    %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
}

// CHECK: func.func @csediv(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.divide %arg0, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.divide %arg1, %arg0 : tensor<2x2xf32>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
// CHECK-NEXT:   return %2 : tensor<2x2xf32>
// CHECK-NEXT: }
