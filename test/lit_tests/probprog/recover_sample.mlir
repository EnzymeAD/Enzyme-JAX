// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  // CPU:       func.func @test_1d_to_scalar(%[[POS:.+]]: tensor<10xf64>) -> tensor<f64> {
  // CPU-NEXT:    %[[SLICE:.+]] = stablehlo.slice %[[POS]] [5:6] : (tensor<10xf64>) -> tensor<1xf64>
  // CPU-NEXT:    %[[RESHAPE:.+]] = stablehlo.reshape %[[SLICE]] : (tensor<1xf64>) -> tensor<f64>
  // CPU-NEXT:    return %[[RESHAPE]] : tensor<f64>
  // CPU-NEXT:  }
  func.func @test_1d_to_scalar(%position: tensor<10xf64>) -> tensor<f64> {
    %result = enzyme.recover_sample %position[5] : tensor<10xf64> -> tensor<f64>
    return %result : tensor<f64>
  }

  // CPU:       func.func @test_1d_to_1d(%[[POS:.+]]: tensor<10xf64>) -> tensor<3xf64> {
  // CPU-NEXT:    %[[SLICE:.+]] = stablehlo.slice %[[POS]] [2:5] : (tensor<10xf64>) -> tensor<3xf64>
  // CPU-NEXT:    %[[RESHAPE:.+]] = stablehlo.reshape %[[SLICE]] : (tensor<3xf64>) -> tensor<3xf64>
  // CPU-NEXT:    return %[[RESHAPE]] : tensor<3xf64>
  // CPU-NEXT:  }
  func.func @test_1d_to_1d(%position: tensor<10xf64>) -> tensor<3xf64> {
    %result = enzyme.recover_sample %position[2] : tensor<10xf64> -> tensor<3xf64>
    return %result : tensor<3xf64>
  }

  // CPU:       func.func @test_2d_batched(%[[POS:.+]]: tensor<5x10xf64>) -> tensor<5x3xf64> {
  // CPU-NEXT:    %[[SLICE:.+]] = stablehlo.slice %[[POS]] [0:5, 2:5] : (tensor<5x10xf64>) -> tensor<5x3xf64>
  // CPU-NEXT:    %[[RESHAPE:.+]] = stablehlo.reshape %[[SLICE]] : (tensor<5x3xf64>) -> tensor<5x3xf64>
  // CPU-NEXT:    return %[[RESHAPE]] : tensor<5x3xf64>
  // CPU-NEXT:  }
  func.func @test_2d_batched(%position: tensor<5x10xf64>) -> tensor<5x3xf64> {
    %result = enzyme.recover_sample %position[2] : tensor<5x10xf64> -> tensor<5x3xf64>
    return %result : tensor<5x3xf64>
  }
}
