// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=broadcast_reduce" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=TD
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s --check-prefix=OPT

module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %cst = stablehlo.constant dense<0.000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<2x3xf32>) -> tensor<2x5x3x4xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2, 1] : (tensor<2x5x3x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }

  // Issue #1788: fuse reshape into reduce and hoist reduce before broadcast_in_dim.
  func.func @reduce_reshape_broadcast_issue1788(%arg0: tensor<128x2x1xf32>) -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2, 3] : (tensor<128x2x1xf32>) -> tensor<128x128x2x1xf32>
    %1 = stablehlo.reshape %0 : (tensor<128x128x2x1xf32>) -> tensor<128x256x1xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x1xf32>, tensor<f32>) -> tensor<128x1xf32>
    return %2 : tensor<128x1xf32>
  }

  func.func @reduce_broadcast_max(%arg0: tensor<16x4xf32>) -> tensor<8x16xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<16x4xf32>) -> tensor<8x16x4xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x16x4xf32>, tensor<f32>) -> tensor<8x16xf32>
    return %1 : tensor<8x16xf32>
  }
}

// TD-LABEL: func.func @main(
// TD-SAME: %[[ARG:.+]]: tensor<2x3xf32>) -> tensor<2x4xf32> {
// TD-DAG: %[[CST5:.+]] = stablehlo.constant dense<5> : tensor<2x4xi64>
// TD-DAG: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// TD-NEXT: %[[REDUC:.+]] = stablehlo.reduce(%[[ARG]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
// TD-NEXT: %[[BROAD:.+]] = stablehlo.broadcast_in_dim %[[REDUC]], dims = [0] : (tensor<2xf32>) -> tensor<2x4xf32>
// TD-NEXT: %[[CONVE:.+]] = stablehlo.convert %[[CST5]] : (tensor<2x4xi64>) -> tensor<2x4xf32>
// TD-NEXT: %[[MULTI:.+]] = stablehlo.multiply %[[BROAD]], %[[CONVE]] : tensor<2x4xf32>
// TD-NEXT: return %[[MULTI]] : tensor<2x4xf32>

// OPT-LABEL: func.func @main(
// OPT-SAME: %[[OARG:.+]]: tensor<2x3xf32>) -> tensor<2x4xf32> {
// OPT-DAG: %[[OCST5:.+]] = stablehlo.constant dense<5.000000e+00> : tensor<2xf32>
// OPT-DAG: %[[OZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// OPT-NEXT: %[[OREDUC:.+]] = stablehlo.reduce(%[[OARG]] init: %[[OZERO]]) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
// OPT-NEXT: %[[OMULTI:.+]] = stablehlo.multiply %[[OREDUC]], %[[OCST5]] : tensor<2xf32>
// OPT-NEXT: %[[OBROAD:.+]] = stablehlo.broadcast_in_dim %[[OMULTI]], dims = [0] : (tensor<2xf32>) -> tensor<2x4xf32>
// OPT-NEXT: return %[[OBROAD]] : tensor<2x4xf32>

// OPT-LABEL: func.func @reduce_reshape_broadcast_issue1788(
// OPT-SAME: %[[IARG:.+]]: tensor<128x2x1xf32>) -> tensor<128x1xf32> {
// OPT-NEXT: %[[IZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// OPT-NEXT: %[[IREDUC:.+]] = stablehlo.reduce(%[[IARG]] init: %[[IZERO]]) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x2x1xf32>, tensor<f32>) -> tensor<1xf32>
// OPT-NEXT: %[[IBROAD:.+]] = stablehlo.broadcast_in_dim %[[IREDUC]], dims = [1] : (tensor<1xf32>) -> tensor<128x1xf32>
// OPT-NEXT: return %[[IBROAD]] : tensor<128x1xf32>

// TD-LABEL: func.func @reduce_broadcast_max(
// TD-SAME: %[[CARG:.+]]: tensor<16x4xf32>) -> tensor<8x16xf32> {
// TD-NEXT: %[[CONE:.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// TD-NEXT: %[[CRED:.+]] = stablehlo.reduce(%[[CARG]] init: %[[CONE]]) applies stablehlo.maximum across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
// TD-NEXT: %[[CBROAD:.+]] = stablehlo.broadcast_in_dim %[[CRED]], dims = [1] : (tensor<16xf32>) -> tensor<8x16xf32>
// TD-NEXT: return %[[CBROAD]] : tensor<8x16xf32>
