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

func.func @main5(%arg0: tensor<10xf64>) -> tensor<10xf64> {
    %cst1 = stablehlo.constant dense<5.0> : tensor<10xf64>
    %cst2 = stablehlo.constant dense<2.0> : tensor<10xf64>
    %0 = stablehlo.multiply %arg0, %cst1 : tensor<10xf64>
    %1 = stablehlo.multiply %cst2, %0 : tensor<10xf64>
    return %1 : tensor<10xf64>
}

// CHECK: func.func @main5(%arg0: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+01> : tensor<10xf64>
// CHECK-NEXT:     %0 = stablehlo.multiply %cst, %arg0 : tensor<10xf64>
// CHECK-NEXT:     return %0 : tensor<10xf64>
// CHECK-NEXT: }

func.func @main6(%arg0: tensor<10xf64>) -> tensor<10xf64> {
    %cst1 = stablehlo.constant dense<5.0> : tensor<10xf64>
    %cst2 = stablehlo.constant dense<2.0> : tensor<10xf64>
    %0 = stablehlo.multiply %cst1, %arg0 : tensor<10xf64>
    %1 = stablehlo.multiply %cst2, %0 : tensor<10xf64>
    return %1 : tensor<10xf64>
}

// CHECK: func.func @main6(%arg0: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+01> : tensor<10xf64>
// CHECK-NEXT:     %0 = stablehlo.multiply %cst, %arg0 : tensor<10xf64>
// CHECK-NEXT:     return %0 : tensor<10xf64>
// CHECK-NEXT: }

func.func @main7(%arg0: tensor<10xf64>) -> tensor<10xf64> {
    %cst1 = stablehlo.constant dense<5.0> : tensor<10xf64>
    %cst2 = stablehlo.constant dense<2.0> : tensor<10xf64>
    %0 = stablehlo.multiply %arg0, %cst1 : tensor<10xf64>
    %1 = stablehlo.multiply %0, %cst2 : tensor<10xf64>
    return %1 : tensor<10xf64>
}

// CHECK: func.func @main7(%arg0: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+01> : tensor<10xf64>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %cst : tensor<10xf64>
// CHECK-NEXT:     return %0 : tensor<10xf64>
// CHECK-NEXT: }

func.func @main8(%arg0: tensor<10xf64>) -> tensor<10xf64> {
    %cst1 = stablehlo.constant dense<5.0> : tensor<10xf64>
    %cst2 = stablehlo.constant dense<2.0> : tensor<10xf64>
    %0 = stablehlo.multiply %cst1, %arg0 : tensor<10xf64>
    %1 = stablehlo.multiply %0, %cst2 : tensor<10xf64>
    return %1 : tensor<10xf64>
}

// CHECK: func.func @main8(%arg0: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+01> : tensor<10xf64>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %cst : tensor<10xf64>
// CHECK-NEXT:     return %0 : tensor<10xf64>
// CHECK-NEXT: }
