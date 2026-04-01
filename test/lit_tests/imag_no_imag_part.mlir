// RUN: enzymexlamlir-opt %s -enzyme-hlo-opt | FileCheck %s

module {
  func.func @test(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %cst = stablehlo.constant dense<[(1.00e+00, 0.00e+00), (2.00e+00, 0.00e+00)]> : tensor<2xcomplex<f32>>
    %cst2 = stablehlo.constant dense<0.00e+00> : tensor<2xf32>
    %0 = stablehlo.complex %arg0, %cst2 : tensor<2xcomplex<f32>>
    %1 = stablehlo.add %0, %cst : tensor<2xcomplex<f32>>
    %2 = stablehlo.imag %1 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    return %2 : tensor<2xf32>
  }
}

// CHECK: func.func @test(%arg0: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT:     return %cst : tensor<2xf32>
// CHECK-NEXT: }
