// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%4273: tensor<1520x3056xf32>, %4258: tensor<1519x3056xf32>) -> (tensor<1x1520x3056xf32>) {

    %4275 = stablehlo.slice %4273 [0:1, 0:3056] : (tensor<1520x3056xf32>) -> tensor<1x3056xf32>

    %4276 = stablehlo.slice %4273 [1:1520, 0:3056] : (tensor<1520x3056xf32>) -> tensor<1519x3056xf32>

    %4278 = stablehlo.reshape %4275 : (tensor<1x3056xf32>) -> tensor<1x1x3056xf32>

    %4277 = stablehlo.add %4276, %4258 : tensor<1519x3056xf32>
    
    %4279 = stablehlo.reshape %4277 : (tensor<1519x3056xf32>) -> tensor<1x1519x3056xf32>


    %4280 = stablehlo.concatenate %4278, %4279, dim = 1 : (tensor<1x1x3056xf32>, tensor<1x1519x3056xf32>) -> tensor<1x1520x3056xf32>
    return %4280 : tensor<1x1520x3056xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1520x3056xf32>, %arg1: tensor<1519x3056xf32>) -> tensor<1x1520x3056xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.pad %arg1, %cst, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1519x3056xf32>, tensor<f32>) -> tensor<1520x3056xf32>
// CHECK-NEXT:    %1 = stablehlo.add %arg0, %0 : tensor<1520x3056xf32>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1520x3056xf32>) -> tensor<1x1520x3056xf32>
// CHECK-NEXT:    return %2 : tensor<1x1520x3056xf32>
// CHECK-NEXT:  }

