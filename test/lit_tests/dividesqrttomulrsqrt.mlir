// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s
module {
  func.func @t1(%arg0: tensor<4x1x3x1xf32>) -> tensor<4x1x3x1xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<4x1x3x1xf32>
    %0 = stablehlo.sqrt %arg0 : tensor<4x1x3x1xf32>
    %1 = stablehlo.divide %cst, %0 : tensor<4x1x3x1xf32>
    return %1 : tensor<4x1x3x1xf32>
  }
}

// CHECK:  func.func @t1(%arg0: tensor<4x1x3x1xf32>) -> tensor<4x1x3x1xf32> {
// CHECK-NEXT:    %0 = stablehlo.rsqrt %arg0 : tensor<4x1x3x1xf32>
// CHECK-NEXT:    return %0 : tensor<4x1x3x1xf32>
// CHECK-NEXT:  }

module {
  func.func @t2(%arg0: tensor<4x1x3x1xf32>) -> tensor<4x1x3x1xf32> {
    %cst = stablehlo.constant dense<5.000000e+00> : tensor<4x1x3x1xf32>
    %0 = stablehlo.sqrt %arg0 : tensor<4x1x3x1xf32>
    %1 = stablehlo.divide %cst, %0 : tensor<4x1x3x1xf32>
    return %1 : tensor<4x1x3x1xf32>
  }
}

// CHECK:  func.func @t2(%arg0: tensor<4x1x3x1xf32>) -> tensor<4x1x3x1xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<5.000000e+00> : tensor<4x1x3x1xf32>
// CHECK-NEXT:    %0 = stablehlo.rsqrt %arg0 : tensor<4x1x3x1xf32>
// CHECK-NEXT:    %1 = stablehlo.multiply %cst, %0 : tensor<4x1x3x1xf32>
// CHECK-NEXT:    return %1 : tensor<4x1x3x1xf32>
// CHECK-NEXT:  }
