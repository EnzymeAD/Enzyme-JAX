// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt)" | FileCheck %s

module {
  func.func @lgamma_f32() -> tensor<f32> {  
    %arg = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = chlo.lgamma %arg : tensor<f32> -> tensor<f32>
    func.return %1 : tensor<f32>
  }
}


// CHECK: func.func @lgamma_f32() -> tensor<f32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<4.76837158E-7> : tensor<f32>
// CHECK-NEXT:   return %cst : tensor<f32>
// CHECK-NEXT: }
