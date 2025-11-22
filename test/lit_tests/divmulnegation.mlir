// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = stablehlo.negate %arg2 : tensor<8xf32>
  %2 = stablehlo.subtract %1, %arg1 : tensor<8xf32>
  %3 = stablehlo.divide %0, %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK: func.func @main1(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NEXT:   %0 = stablehlo.negate %arg2 : tensor<8xf32>
// CHECK-NEXT:   %1 = stablehlo.subtract %arg1, %0 : tensor<8xf32>
// CHECK-NEXT:   %2 = stablehlo.divide %arg0, %1 : tensor<8xf32>
// CHECK-NEXT:   return %2 : tensor<8xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = stablehlo.negate %arg2 : tensor<8xf32>
  %2 = stablehlo.subtract %1, %arg1 : tensor<8xf32>
  %3 = stablehlo.multiply %0, %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK: func.func @main2(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NEXT:   %0 = stablehlo.negate %arg2 : tensor<8xf32>
// CHECK-NEXT:   %1 = stablehlo.subtract %arg1, %0 : tensor<8xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %arg0, %1 : tensor<8xf32>
// CHECK-NEXT:   return %2 : tensor<8xf32>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = stablehlo.negate %arg2 : tensor<8xf32>
  %2 = stablehlo.subtract %1, %arg1 : tensor<8xf32>
  %3 = stablehlo.divide %2, %0 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK: func.func @main3(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NEXT:   %0 = stablehlo.negate %arg2 : tensor<8xf32>
// CHECK-NEXT:   %1 = stablehlo.subtract %arg1, %0 : tensor<8xf32>
// CHECK-NEXT:   %2 = stablehlo.divide %1, %arg0 : tensor<8xf32>
// CHECK-NEXT:   return %2 : tensor<8xf32>
// CHECK-NEXT: }

func.func @main4(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = stablehlo.negate %arg2 : tensor<8xf32>
  %2 = stablehlo.subtract %1, %arg1 : tensor<8xf32>
  %3 = stablehlo.multiply %2, %0 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK: func.func @main4(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NEXT:   %0 = stablehlo.negate %arg2 : tensor<8xf32>
// CHECK-NEXT:   %1 = stablehlo.subtract %arg1, %0 : tensor<8xf32>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %arg0 : tensor<8xf32>
// CHECK-NEXT:   return %2 : tensor<8xf32>
// CHECK-NEXT: }
