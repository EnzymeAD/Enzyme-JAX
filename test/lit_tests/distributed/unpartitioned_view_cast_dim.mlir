// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s -o /dev/null

// Building a matrix out of scalars concatenates along dimensions that must stay
// local. Their view casts and anchors carry those dimensions as device-local
// axes, which the axis analysis ClusterDistributedKernels rebuilds must read as
// unshardable symbols of the dimension's size, or they cannot merge with the
// consuming op's symbol for the same dimension.
module @rot_matrix {
  sdy.mesh @mesh = <["x"=2]>

  func.func @main(%c: tensor<f32>, %s: tensor<f32>) -> tensor<2x2xf32> {
    %c1 = stablehlo.reshape %c : (tensor<f32>) -> tensor<1xf32>
    %s1 = stablehlo.reshape %s : (tensor<f32>) -> tensor<1xf32>
    %row = stablehlo.concatenate %c1, %s1, dim = 0
        : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %row2 = stablehlo.reshape %row : (tensor<2xf32>) -> tensor<1x2xf32>
    %m = stablehlo.concatenate %row2, %row2, dim = 0
        : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    return %m : tensor<2x2xf32>
  }
}
