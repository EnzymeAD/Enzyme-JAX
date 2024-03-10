// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<2x2xf32>) -> tensor<2x2xf32> {
    %pv = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %a, %pv, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    return %pad : tensor<2x2xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2x2xf32>
// CHECK-NEXT:    return %arg0 : tensor<2x2xf32>
// CHECK-NEXT:  }
