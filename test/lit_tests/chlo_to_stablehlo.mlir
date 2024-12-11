// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {

func.func @lgamma_f32() -> tensor<f32> {  
  %arg = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = chlo.lgamma %arg : tensor<f32> -> tensor<f32>
  func.return %1 : tensor<f32>
}

}
