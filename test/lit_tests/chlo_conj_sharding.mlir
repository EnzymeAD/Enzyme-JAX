// RUN: enzymexlamlir-opt --sdy-basic-propagate %s | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func.func @conj
func.func @conj(%arg0: tensor<2x4x32x32xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {"a"}, {}, {}]>}) -> tensor<2x4x32x32xcomplex<f64>> {
  // CHECK: chlo.conj %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", ?}, {"a", ?}, {?}, {?}]>]>}
  %0 = chlo.conj %arg0 : tensor<2x4x32x32xcomplex<f64>> -> tensor<2x4x32x32xcomplex<f64>>
  return %0 : tensor<2x4x32x32xcomplex<f64>>
}
