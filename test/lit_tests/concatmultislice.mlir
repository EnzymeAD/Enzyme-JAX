// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%11825: tensor<4x5x79xf64>, %11829 : tensor<4x5x1xf64>, %cst_286 : tensor<f64>, %z : tensor<4x6x1xf64>) -> tensor<4x6x80xf64> {
      %11826 = stablehlo.pad %11825, %cst_286, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x5x79xf64>, tensor<f64>) -> tensor<4x6x79xf64>
      %11830 = stablehlo.pad %11829, %cst_286, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x5x1xf64>, tensor<f64>) -> tensor<4x6x1xf64>
      %11832 = stablehlo.concatenate %11826, %11830, %z, dim = 2 : (tensor<4x6x79xf64>, tensor<4x6x1xf64>, tensor<4x6x1xf64>) -> tensor<4x6x81xf64>
      stablehlo.return %11832 : tensor<4x6x81xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x5x79xf64>, %arg1: tensor<4x5x1xf64>, %arg2: tensor<f64>, %arg3: tensor<4x6x1xf64>) -> tensor<4x6x80xf64> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 2 : (tensor<4x5x79xf64>, tensor<4x5x1xf64>) -> tensor<4x5x80xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg2, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x5x80xf64>, tensor<f64>) -> tensor<4x6x80xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, %arg3, dim = 2 : (tensor<4x6x80xf64>, tensor<4x6x1xf64>) -> tensor<4x6x81xf64>
// CHECK-NEXT:    stablehlo.return %2 : tensor<4x6x81xf64>
// CHECK-NEXT:  }