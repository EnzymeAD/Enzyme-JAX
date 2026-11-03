// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// A non-negative f64 latency is accepted.
// CHECK-LABEL: func.func @latency_accepted
// CHECK: stablehlo.add {{.*}} {distributed.latency = 2.500000e+00 : f64}
func.func @latency_accepted(%x: tensor<4xf32>) -> tensor<4xf32> {
  %s = stablehlo.add %x, %x {distributed.latency = 2.5 : f64} : tensor<4xf32>
  return %s : tensor<4xf32>
}

// -----

// A negative latency is rejected.
func.func @negative_latency_below_zero(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @below {{expects 'distributed.latency' to be a non-negative f64 number of nanoseconds}}
  %s = stablehlo.add %x, %x {distributed.latency = -1.0 : f64} : tensor<4xf32>
  return %s : tensor<4xf32>
}

// -----

// A latency that is not a float is rejected.
func.func @negative_latency_not_float(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @below {{expects 'distributed.latency' to be a non-negative f64 number of nanoseconds}}
  %s = stablehlo.add %x, %x {distributed.latency = 3 : i64} : tensor<4xf32>
  return %s : tensor<4xf32>
}
