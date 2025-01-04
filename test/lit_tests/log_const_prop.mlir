// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt)" | FileCheck %s

module {
  func.func @log_f32() -> tensor<2xf32> {  
    %arg = stablehlo.constant dense<[1.000000e+00,2.000000e+00]> : tensor<2xf32>
    %result = stablehlo.log %arg : tensor<2xf32> 
    func.return %result : tensor<2xf32>
  }

  func.func @log_plus_one_op_test_f64() -> tensor<5xf64> {
    %operand = stablehlo.constant dense<[0.0, -0.999, 7.0, 6.38905621, 15.0]> : tensor<5xf64>
    %result = stablehlo.log_plus_one %operand : tensor<5xf64>
    func.return %result : tensor<5xf64>
  }
}
