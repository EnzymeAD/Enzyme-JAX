// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-enzyme-probprog{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  func.func private @model.simulate() -> !enzyme.Trace

  func.func @test() -> tensor<1xui64> {
    %0 = func.call @model.simulate() : () -> !enzyme.Trace
    %1 = builtin.unrealized_conversion_cast %0 : !enzyme.Trace to tensor<1xui64>
    return %1 : tensor<1xui64>
  }
}

// CPU:  func.func private @model.simulate() -> tensor<1xui64>
// CPU-NEXT:  func.func @test() -> tensor<1xui64> {
// CPU-NEXT:    %0 = call @model.simulate() : () -> tensor<1xui64>
// CPU-NEXT:    return %0 : tensor<1xui64>
// CPU-NEXT:  }