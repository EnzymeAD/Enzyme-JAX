// Regression test: stablehlo.slice used for static per-layer indexing (e.g.
// `weights[i, :, :]` inside an unrolled Python loop over layers) gets a
// Shardy sharding rule with a "permutation" factor for the sliced-away
// dimension. Such a factor would need a collective-permute if it were ever
// actually sharded, which isn't implemented here, so the axis analysis
// instead marks that dimension unshardable: it still unifies normally with
// whatever it would otherwise merge with, but never gets anchored to a real
// logical/physical sharding axis. This should compile cleanly instead of
// asserting or, in a release build with asserts compiled out, silently
// computing a 0-extent local dimension. (A debug build also emits an
// informational remark here, but that's suppressed in release builds along
// with the rest of this analysis's per-op diagnostics, so it isn't checked
// below.)
// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards --convert-main-to-distributed-function --materialize-distributed-collectives %s | FileCheck %s

// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s | FileCheck %s --check-prefix=FULL

// The unshardable dimension is materialized as a device-local axis, sized to
// each tensor separately: 3 on the slice's operand and 1 on its result, which
// only exists once kernels are clustered.
// CHECK-NOT: distributed.LogicalMeshAxes 3
// CHECK: distributed.DeviceLocalAxis 3
// CHECK: distributed.DistributedYield
// FULL: distributed.DeviceLocalAxis 3
// FULL: distributed.DeviceLocalAxis 1

module @permutation_factor_slice {
  sdy.mesh @mesh = <["tp"=4]>

  func.func @main(
      %arg0: tensor<3x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"tp"}, {}]>},
      %arg1: tensor<8xf32>)
      -> (tensor<8xf32>) {
    // Static index 0 picks one of the 3 stacked layers; Shardy tags this
    // slice's dim-0 factor (extent 3) as a permutation factor.
    %w0 = stablehlo.slice %arg0 [0:1, 0:8, 0:8] : (tensor<3x8x8xf32>) -> tensor<1x8x8xf32>
    %w = stablehlo.reshape %w0 : (tensor<1x8x8xf32>) -> tensor<8x8xf32>
    %out = stablehlo.dot_general %w, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] :
      (tensor<8x8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %out : tensor<8xf32>
  }
}
