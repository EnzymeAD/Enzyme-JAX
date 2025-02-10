// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<10xf64>
    %1 = stablehlo.add %arg1, %0 : tensor<10xf64>
    %2 = stablehlo.add %arg0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK:  func.func @main1(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg1 : tensor<10xf64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %0 : tensor<10xf64>
// CHECK-NEXT:    return %1 : tensor<10xf64>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %0 = stablehlo.add %arg1, %arg0 : tensor<10xf64>
    %1 = stablehlo.add %0, %arg1 : tensor<10xf64>
    %2 = stablehlo.add %arg0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK:  func.func @main2(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg1, %arg0 : tensor<10xf64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %0 : tensor<10xf64>
// CHECK-NEXT:    return %1 : tensor<10xf64>
// CHECK-NEXT:  }

func.func @main3(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %0 = stablehlo.multiply %arg1, %arg0 : tensor<10xf64>
    %1 = stablehlo.multiply %0, %arg1 : tensor<10xf64>
    %2 = stablehlo.multiply %arg0, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK:  func.func @main3(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg1, %arg0 : tensor<10xf64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<10xf64>
// CHECK-NEXT:    return %1 : tensor<10xf64>
// CHECK-NEXT:  }

func.func @main4(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
    %0 = stablehlo.multiply %arg1, %arg0 : tensor<10xf64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<10xf64>
    %2 = stablehlo.multiply %arg1, %1 : tensor<10xf64>
    return %2 : tensor<10xf64>
}

// CHECK:  func.func @main4(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg1, %arg0 : tensor<10xf64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<10xf64>
// CHECK-NEXT:    return %1 : tensor<10xf64>
// CHECK-NEXT:  }
