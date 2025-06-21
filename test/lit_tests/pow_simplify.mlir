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

func.func @main7(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x4xf32>
    %0 = stablehlo.power %cst, %arg0 : tensor<10x4xf32>
    return %0 : tensor<10x4xf32>
}

// CHECK: func.func @main7(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<10x4xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<10x4xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<10x4xf32>
// CHECK-NEXT:     %0 = stablehlo.compare  GT, %arg0, %cst_1 : (tensor<10x4xf32>, tensor<10x4xf32>) -> tensor<10x4xi1>
// CHECK-NEXT:     %1 = stablehlo.select %0, %cst_1, %cst_0 : tensor<10x4xi1>, tensor<10x4xf32>
// CHECK-NEXT:     %2 = stablehlo.compare  EQ, %arg0, %cst_1 : (tensor<10x4xf32>, tensor<10x4xf32>) -> tensor<10x4xi1>
// CHECK-NEXT:     %3 = stablehlo.select %2, %cst, %1 : tensor<10x4xi1>, tensor<10x4xf32>
// CHECK-NEXT:     return %3 : tensor<10x4xf32>
// CHECK-NEXT: }
