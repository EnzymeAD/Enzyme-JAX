// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --sdy-insert-explicit-reshards=enable-full-version=true --sdy-reshard-to-collectives --insert-identity-reshard --shardy-to-distributed --localize-distributed-module --cse --distributed-simplify-collectives --distributed-overlap-communication-module %s > /dev/null
// Four batched square matmuls in the chain B(A(C(DA))) where A is reused,
// creating a dependency from the innermost to the third matmul.
//
// Shape:
//   t1 = D @ A        (innermost)
//   t2 = C @ t1
//   t3 = A @ t2       (A reused here)
//   t4 = B @ t3       (outermost)
//
// A, B, C, D are all [batch=8, N=512, N=512] square matrices.
// All inputs are batch-sharded on "x"; contracting dims are unsharded so
// no all-reduce arises from a single config.

sdy.mesh @mesh = <["x"=2]>

distributed.AxisAllToAll @ax1 2 1600000000 0
distributed.PhysicalMesh @phys_mesh [@ax1]

func.func @bacda_batch_matmul(
    %a: tensor<8x512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>},
    %b: tensor<8x512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>},
    %c: tensor<8x512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>},
    %d: tensor<8x512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}
) -> (tensor<8x512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) {
  // t1 = D @ A
  %t1 = stablehlo.dot_general %d, %a,
      batching_dims = [0] x [0], contracting_dims = [2] x [1]
      : (tensor<8x512x512xf32>, tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
  // t2 = C @ t1
  %t2 = stablehlo.dot_general %c, %t1,
      batching_dims = [0] x [0], contracting_dims = [2] x [1]
      : (tensor<8x512x512xf32>, tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
  // t3 = A @ t2  (A reused)
  %t3 = stablehlo.dot_general %a, %t2,
      batching_dims = [0] x [0], contracting_dims = [2] x [1]
      : (tensor<8x512x512xf32>, tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
  // t4 = B @ t3
  %t4 = stablehlo.dot_general %b, %t3,
      batching_dims = [0] x [0], contracting_dims = [2] x [1]
      : (tensor<8x512x512xf32>, tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
  return %t4 : tensor<8x512x512xf32>
}
