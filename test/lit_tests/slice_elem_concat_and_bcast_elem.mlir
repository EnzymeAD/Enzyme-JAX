// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=slice_elementwise<1>;slice_concat<1>;noop_slice<16>},transform-interpreter,enzyme-hlo-remove-transform,canonicalize,cse)" | FileCheck %s --check-prefix=TD
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=65536},enzyme-hlo-opt{passses=548863})" | FileCheck %s --check-prefix=OPT

module {
  // TD-LABEL: func.func @full_coverage_slice_elementwise
  // TD-NOT: stablehlo.concatenate
  // TD-NOT: stablehlo.slice
  // TD-DAG: %[[R0:.*]] = stablehlo.add %arg0, %arg0 : tensor<1x4x4xf32>
  // TD-DAG: %[[R1:.*]] = stablehlo.add %arg1, %arg1 : tensor<1x4x4xf32>
  // TD: return %[[R0]], %[[R1]]
  //
  // OPT-LABEL: func.func @full_coverage_slice_elementwise
  // OPT-NOT: stablehlo.concatenate
  // OPT-NOT: stablehlo.slice
  // OPT-DAG: %[[OR0:.*]] = stablehlo.multiply %arg0, %{{.*}} : tensor<1x4x4xf32>
  // OPT-DAG: %[[OR1:.*]] = stablehlo.multiply %arg1, %{{.*}} : tensor<1x4x4xf32>
  // OPT: return %[[OR0]], %[[OR1]]
  func.func @full_coverage_slice_elementwise(
      %lane0: tensor<1x4x4xf32>,
      %lane1: tensor<1x4x4xf32>)
      -> (tensor<1x4x4xf32>, tensor<1x4x4xf32>) {
    %packed = stablehlo.concatenate %lane0, %lane1, dim = 0
      : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
    %sum = stablehlo.add %packed, %packed : tensor<2x4x4xf32>
    %result0 = stablehlo.slice %sum [0:1, 0:4, 0:4]
      : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
    %result1 = stablehlo.slice %sum [1:2, 0:4, 0:4]
      : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
    return %result0, %result1
      : tensor<1x4x4xf32>, tensor<1x4x4xf32>
  }

  // TD-LABEL: func.func @shared_tangent_8_lanes
  // TD-NOT: stablehlo.concatenate
  // TD-NOT: stablehlo.slice
  // TD: return
  //
  // OPT-LABEL: func.func @shared_tangent_8_lanes
  // OPT-NOT: stablehlo.concatenate
  // OPT-NOT: stablehlo.slice
  // OPT: return
  func.func @shared_tangent_8_lanes(
      %lhs0: tensor<1x16x16xf64>, %lhs1: tensor<1x16x16xf64>,
      %lhs2: tensor<1x16x16xf64>, %lhs3: tensor<1x16x16xf64>,
      %lhs4: tensor<1x16x16xf64>, %lhs5: tensor<1x16x16xf64>,
      %lhs6: tensor<1x16x16xf64>, %lhs7: tensor<1x16x16xf64>,
      %rhs0: tensor<1x16x16xf64>, %rhs1: tensor<1x16x16xf64>,
      %rhs2: tensor<1x16x16xf64>, %rhs3: tensor<1x16x16xf64>,
      %rhs4: tensor<1x16x16xf64>, %rhs5: tensor<1x16x16xf64>,
      %rhs6: tensor<1x16x16xf64>, %rhs7: tensor<1x16x16xf64>)
      -> (tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>) {
    %lhs = stablehlo.concatenate
        %lhs0, %lhs1, %lhs2, %lhs3, %lhs4, %lhs5, %lhs6, %lhs7, dim = 0
        : (tensor<1x16x16xf64>, tensor<1x16x16xf64>,
           tensor<1x16x16xf64>, tensor<1x16x16xf64>,
           tensor<1x16x16xf64>, tensor<1x16x16xf64>,
           tensor<1x16x16xf64>, tensor<1x16x16xf64>)
        -> tensor<8x16x16xf64>
    %rhs = stablehlo.concatenate
        %rhs0, %rhs1, %rhs2, %rhs3, %rhs4, %rhs5, %rhs6, %rhs7, dim = 0
        : (tensor<1x16x16xf64>, tensor<1x16x16xf64>,
           tensor<1x16x16xf64>, tensor<1x16x16xf64>,
           tensor<1x16x16xf64>, tensor<1x16x16xf64>,
           tensor<1x16x16xf64>, tensor<1x16x16xf64>)
        -> tensor<8x16x16xf64>

    %shared = stablehlo.multiply %lhs, %rhs : tensor<8x16x16xf64>
    %sum = stablehlo.add %lhs, %shared : tensor<8x16x16xf64>
    %difference = stablehlo.subtract %rhs, %shared : tensor<8x16x16xf64>

    %sum0 = stablehlo.slice %sum [0:1, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %sum1 = stablehlo.slice %sum [1:2, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %sum2 = stablehlo.slice %sum [2:3, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %sum3 = stablehlo.slice %sum [3:4, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %sum4 = stablehlo.slice %sum [4:5, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %sum5 = stablehlo.slice %sum [5:6, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %sum6 = stablehlo.slice %sum [6:7, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %sum7 = stablehlo.slice %sum [7:8, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>

    %difference0 = stablehlo.slice %difference [0:1, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %difference1 = stablehlo.slice %difference [1:2, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %difference2 = stablehlo.slice %difference [2:3, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %difference3 = stablehlo.slice %difference [3:4, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %difference4 = stablehlo.slice %difference [4:5, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %difference5 = stablehlo.slice %difference [5:6, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %difference6 = stablehlo.slice %difference [6:7, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>
    %difference7 = stablehlo.slice %difference [7:8, 0:16, 0:16] : (tensor<8x16x16xf64>) -> tensor<1x16x16xf64>

    return %sum0, %sum1, %sum2, %sum3, %sum4, %sum5, %sum6, %sum7,
        %difference0, %difference1, %difference2, %difference3,
        %difference4, %difference5, %difference6, %difference7
        : tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>,
          tensor<1x16x16xf64>, tensor<1x16x16xf64>
  }

  // OPT-LABEL: func.func @broadcast_transpose_reshape_rotate
  // OPT-NOT: stablehlo.transpose
  // OPT-NOT: stablehlo.reshape
  // OPT-NOT: stablehlo.concatenate
  // OPT-NOT: stablehlo.broadcast_in_dim
  // OPT: %[[ROT:.*]] = "enzymexla.rotate"(%arg0) <{amount = 12271 : i32, dimension = 2 : i32}> : (tensor<1x6128x12272xf64>) -> tensor<1x6128x12272xf64>
  // OPT: %[[SUB:.*]] = stablehlo.subtract %arg1, %[[ROT]] : tensor<1x6128x12272xf64>
  // OPT: %[[MUL1:.*]] = stablehlo.multiply %[[SUB]], %arg2 : tensor<1x6128x12272xf64>
  // OPT: %[[MUL2:.*]] = stablehlo.multiply %{{.*}}, %[[MUL1]] : tensor<1x6128x12272xf64>
  // OPT: return %[[MUL2]]
  func.func @broadcast_transpose_reshape_rotate(
      %arg0: tensor<1x6128x12272xf64>,
      %arg1: tensor<1x6128x12272xf64>,
      %arg2: tensor<1x6128x12272xf64>) -> tensor<1x6128x12272xf64> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<12272x6128xf64>
    %t1 = stablehlo.transpose %arg1, dims = [0, 2, 1] : (tensor<1x6128x12272xf64>) -> tensor<1x12272x6128xf64>
    %r1 = stablehlo.reshape %t1 : (tensor<1x12272x6128xf64>) -> tensor<12272x6128xf64>
    %t2 = stablehlo.transpose %arg2, dims = [0, 2, 1] : (tensor<1x6128x12272xf64>) -> tensor<1x12272x6128xf64>
    %r2 = stablehlo.reshape %t2 : (tensor<1x12272x6128xf64>) -> tensor<12272x6128xf64>

    %s0 = stablehlo.slice %arg0 [0:1, 0:6128, 12271:12272] : (tensor<1x6128x12272xf64>) -> tensor<1x6128x1xf64>
    %rs0 = stablehlo.reshape %s0 : (tensor<1x6128x1xf64>) -> tensor<1x6128xf64>

    %t0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<1x6128x12272xf64>) -> tensor<1x12272x6128xf64>
    %rt0 = stablehlo.reshape %t0 : (tensor<1x12272x6128xf64>) -> tensor<12272x6128xf64>

    %s1 = stablehlo.slice %rt0 [0:12271, 0:6128] : (tensor<12272x6128xf64>) -> tensor<12271x6128xf64>
    %cat = stablehlo.concatenate %rs0, %s1, dim = 0 : (tensor<1x6128xf64>, tensor<12271x6128xf64>) -> tensor<12272x6128xf64>

    %sub = stablehlo.subtract %r1, %cat : tensor<12272x6128xf64>
    %mul1 = stablehlo.multiply %sub, %r2 : tensor<12272x6128xf64>
    %mul2 = stablehlo.multiply %cst, %mul1 : tensor<12272x6128xf64>
    %bcast = stablehlo.broadcast_in_dim %mul2, dims = [2, 1] : (tensor<12272x6128xf64>) -> tensor<1x6128x12272xf64>
    return %bcast : tensor<1x6128x12272xf64>
  }
}
