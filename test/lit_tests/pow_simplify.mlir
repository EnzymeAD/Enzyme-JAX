// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="no_nan" | FileCheck %s

func.func @main1(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<10x4xf32>
    %0 = stablehlo.power %arg0, %cst : tensor<10x4xf32>
    return %0 : tensor<10x4xf32>
}

// CHECK: func.func @main1(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg0 : tensor<10x4xf32>
// CHECK-NEXT:     return %0 : tensor<10x4xf32>
// CHECK-NEXT:   }

func.func @main2(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<-1.000000e+00> : tensor<10x4xf32>
    %0 = stablehlo.power %arg0, %cst : tensor<10x4xf32>
    return %0 : tensor<10x4xf32>
}

// CHECK:  func.func @main2(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<10x4xf32>
// CHECK-NEXT:    %0 = stablehlo.divide %cst, %arg0 : tensor<10x4xf32>
// CHECK-NEXT:    return %0 : tensor<10x4xf32>
// CHECK-NEXT:  }

func.func @main3(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x4xf32>
    %0 = stablehlo.power %arg0, %cst : tensor<10x4xf32>
    return %0 : tensor<10x4xf32>
}

// CHECK:  func.func @main3(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<10x4xf32>
// CHECK-NEXT:    return %cst : tensor<10x4xf32>
// CHECK-NEXT:  }

func.func @main4(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<10x4xf32>
    %0 = stablehlo.power %arg0, %cst : tensor<10x4xf32>
    return %0 : tensor<10x4xf32>
}

// CHECK:  func.func @main4(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:    return %arg0 : tensor<10x4xf32>
// CHECK-NEXT:  }

func.func @main5(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<0.500000e+00> : tensor<10x4xf32>
    %0 = stablehlo.power %arg0, %cst : tensor<10x4xf32>
    return %0 : tensor<10x4xf32>
}

// CHECK:  func.func @main5(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.sqrt %arg0 : tensor<10x4xf32>
// CHECK-NEXT:    return %0 : tensor<10x4xf32>
// CHECK-NEXT:  }

func.func @main6(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<-0.500000e+00> : tensor<10x4xf32>
    %0 = stablehlo.power %arg0, %cst : tensor<10x4xf32>
    return %0 : tensor<10x4xf32>
}

// CHECK:  func.func @main6(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.rsqrt %arg0 : tensor<10x4xf32>
// CHECK-NEXT:    return %0 : tensor<10x4xf32>
// CHECK-NEXT:  }


func.func @main7(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %pow = stablehlo.constant dense<[0.0, 0.5, 1.0]> : tensor<3xf32>
    %0 = stablehlo.power %arg0, %pow : tensor<3xf32>
    return %0 : tensor<3xf32>
}

// CHECK:  func.func @main7(%arg0: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<[0.000000e+00, 5.000000e-01, 1.000000e+00]> : tensor<3xf32>
// CHECK-NEXT:    %0 = stablehlo.power %arg0, %cst : tensor<3xf32>
// CHECK-NEXT:    return %0 : tensor<3xf32>
// CHECK-NEXT:  }

func.func @main8(%arg0: tensor<10000x10000xf64>) -> tensor<10000x10000xf64> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<10000x10000xf64>
  %0 = stablehlo.power %arg0, %cst : tensor<10000x10000xf64>
  return %0 : tensor<10000x10000xf64>
}

// CHECK:  func.func @main8(%arg0: tensor<10000x10000xf64>) -> tensor<10000x10000xf64> {
// CHECK-NEXT:    return %arg0 : tensor<10000x10000xf64>
// CHECK-NEXT:  }
