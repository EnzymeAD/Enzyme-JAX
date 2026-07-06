// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-impulse-to-stablehlo{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  // CPU:  func.func @test(%arg0: tensor<10xf64>) -> tensor<10xf64> {
  // CPU-NEXT:    %0 = chlo.lgamma %arg0 : tensor<10xf64>
  // CPU-NEXT:    return %0 : tensor<10xf64>
  // CPU-NEXT:  }
  func.func @test(%x: tensor<10xf64>) -> tensor<10xf64> {
    %result = impulse.lgamma %x : (tensor<10xf64>) -> tensor<10xf64>
    return %result : tensor<10xf64>
  }
}
