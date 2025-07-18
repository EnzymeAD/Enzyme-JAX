// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-enzyme-probprog{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  func.func private @test.simulate(%arg0: tensor<2xui64>) -> (!enzyme.Trace, tensor<f64>)
  
  func.func @simulate(%arg0: tensor<2xui64>) -> (tensor<ui64>, tensor<f64>) {
    %0:2 = call @test.simulate(%arg0) : (tensor<2xui64>) -> (!enzyme.Trace, tensor<f64>)
    %1 = builtin.unrealized_conversion_cast %0#0 : !enzyme.Trace to tensor<ui64>
    return %1, %0#1 : tensor<ui64>, tensor<f64>
  }
}

// CPU:  func.func @simulate(%arg0: tensor<2xui64>) -> (tensor<ui64>, tensor<f64>) {
// CPU-NEXT:    %0:2 = call @test.simulate(%arg0) : (tensor<2xui64>) -> (tensor<ui64>, tensor<f64>)
// CPU-NEXT:    return %0#0, %0#1 : tensor<ui64>, tensor<f64>
// CPU-NEXT:  }
